"""共享 LLM 客户端（OpenAI 兼容网关）。

所有 Agent（文献检索 / 文献审评 / 评审官 / 写作）统一走这里，避免各模块重复
实现网关连接与超时重试逻辑。设计原则：

- 自动从项目根目录 .env 注入网关环境变量（OPENAI_BASE_URL / OPENAI_API_KEY /
  OPENAI_MODEL / LLM_EXTRA_BODY）；已存在的进程环境变量优先（网页注入的用户 Key 优先）。
- 支持备用通道（OPENAI_BASE_URL_2 / OPENAI_API_KEY_2 / OPENAI_MODEL_2 / LLM_EXTRA_BODY_2）：
  主通道连续限流/过载达阈值时自动切换，不必等人来救，也绝不因此降低产出质量。
- available() 返回网关是否真可用（Key 合法 + 至少一次握手成功），调用方据此决定
  走「LM 智能路径」还是「确定性兜底路径」——本系统所有 LM 调用都应有兜底，不能因
  网关挂掉而崩。
- chat_text / chat_json 统一带超时与指数退避重试；网关异常一律抛出，由调用方捕获走兜底。

注意：本模块不缓存任何论文内容，只负责「把 prompt 送到网关、把结果拿回来」。
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── OpenAI 错误类型（防御式导入，缺失时退化为基类） ─────────────────────
try:  # pragma: no cover
    from openai import (
        RateLimitError, APIStatusError, APITimeoutError,
        APIConnectionError, APIError,
    )
except Exception:  # noqa: BLE001
    RateLimitError = APIStatusError = APITimeoutError = APIConnectionError = APIError = Exception


def _load_dotenv() -> None:
    """把项目根 .env 注入环境变量；已存在的进程变量不覆盖（用户 Key 优先）。"""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _real_key(key: str) -> bool:
    if not key or len(key) < 20:
        return False
    if any(ord(c) > 127 for c in key):
        return False
    if key.startswith("在此"):
        return False
    if re.match(r"^(your[_ -]?api[_ -]?key|replace[_ -]?me|<[^>]+>)$", key, re.I):
        return False
    return True


_ACTIVE_IDX = [0]
_FAILOVER_AFTER = int(os.environ.get("LLM_FAILOVER_AFTER", "3") or 3)


def channels() -> list[dict]:
    """列出已配置的通道：主通道 + 可选 _2 / _3 备用。不返回密钥明文供打印。"""
    _load_dotenv()
    default_model = os.environ.get("OPENAI_MODEL", "deepseek-v4-pro")
    out = []
    for suffix in ("", "_2", "_3"):
        base = os.environ.get("OPENAI_BASE_URL" + suffix)
        key = os.environ.get("OPENAI_API_KEY" + suffix)
        if not base or not _real_key(key or ""):
            continue
        try:
            extra = json.loads(os.environ.get("LLM_EXTRA_BODY" + suffix, "{}"))
        except Exception:
            extra = {}
        out.append({
            "base": base, "key": key,
            "model": os.environ.get("OPENAI_MODEL" + suffix) or default_model,
            "extra": extra,
            "label": ("主通道" if not suffix else "备用通道" + suffix),
        })
    return out


def current_channel() -> int:
    chs = channels()
    return (_ACTIVE_IDX[0] % len(chs)) if chs else -1


def _client_or_none():
    """返回 (client, model, extra) 或 None。不抛异常。走当前激活的通道。"""
    chs = channels()
    if not chs:
        return None
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        return None
    ch = chs[_ACTIVE_IDX[0] % len(chs)]
    # max_retries=0：SDK 默认重试 2 次，会和本模块的退避重试叠乘（一次探测变 6 次请求），
    # 网关回 502 时会表现为「卡住 60 秒以上」。重试策略统一由 _raw_chat 控制。
    client = OpenAI(base_url=ch["base"], api_key=ch["key"], max_retries=0)
    return (client, ch["model"], ch["extra"])


_CACHE_AVAIL: bool | None = None
# 「不可用」不能永久缓存：探测只有 6 秒、1 次尝试，网关偶发 429/握手超时就会把它判死。
# 一旦永久置 False，同一个进程里的后续所有调用（评审官、文献精读、正文兜底）全部跳过，
# 表现为「明明网关已恢复，整篇论文还在走模板」。所以负结果只缓存 _NEG_TTL 秒。
_CACHE_AT: float = 0.0
_NEG_TTL: float = 30.0


# ── 全局节流（信号量 + 最小间隔） ────────────────────────────────────────
# 多 Agent 串行调用同一网关，若并发/高频打过去会被「queue.full」限流。这里用进程内
# 互斥锁串起所有 LLM 调用，并保证两次调用之间至少间隔 LLM_MIN_INTERVAL 秒，从源头
# 避免触发网关限流。
_RL_LOCK = threading.Lock()
_RL_LAST = [0.0]
_RL_CALLS = [0]
_RL_MIN_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "1.2"))  # 秒
_RL_MAX_CALLS = int(os.environ.get("LLM_MAX_CALLS", "0") or 0)  # 0=不限

# ── 熔断器（circuit breaker） ────────────────────────────────────────────
# 若网关持续限流/过载，逐个调用空熬退避会拖垮整个流水线。连续失败达阈值即「开路」，
# 此后所有 LLM 调用立即抛错，让各 Agent 秒退到确定性兜底，而不是空转数分钟。
_CB_FAILS = [0]
_CB_OPEN = [False]
_CB_OPEN_UNTIL = [0.0]
_CB_THRESHOLD = int(os.environ.get("LLM_CB_THRESHOLD", "4"))  # 连续失败上限
_CB_COOLDOWN = float(os.environ.get("LLM_CB_COOLDOWN", "120"))  # 开路冷却秒数


def _cb_open() -> bool:
    if not _CB_OPEN[0]:
        return False
    if time.time() >= _CB_OPEN_UNTIL[0]:
        # 冷却结束，允许一次试探
        return False
    return True


def _cb_trip_or_reset(failed: bool) -> None:
    with _RL_LOCK:
        if failed:
            _CB_FAILS[0] += 1
            if _CB_FAILS[0] >= _CB_THRESHOLD:
                _CB_OPEN[0] = True
                _CB_OPEN_UNTIL[0] = time.time() + _CB_COOLDOWN
                print(f"[LLM] 熔断器开路：连续 {_CB_FAILS[0]} 次网关失败，"
                      f"{_CB_COOLDOWN:.0f}s 内强制走确定性兜底")
        else:
            _CB_FAILS[0] = 0
            if _CB_OPEN[0]:
                _CB_OPEN[0] = False
                print("[LLM] 熔断器复位：网关恢复")


def switch_channel() -> bool:
    """切到下一条通道并复位熔断计数；只有一条通道时返回 False（不假装切了）。"""
    chs = channels()
    if len(chs) < 2:
        return False
    _ACTIVE_IDX[0] = (_ACTIVE_IDX[0] + 1) % len(chs)
    _CB_FAILS[0] = 0
    _CB_OPEN[0] = False
    ch = chs[_ACTIVE_IDX[0]]
    print(f"[LLM] 已切到{ch['label']}：{ch['base']} / {ch['model']}")
    return True


def set_min_interval(sec: float) -> None:
    """调小/调大两次调用之间的最小间隔（默认 1.2s；限流严重时调到 1.5~2.5）。"""
    global _RL_MIN_INTERVAL
    _RL_MIN_INTERVAL = max(0.0, float(sec))


def calls() -> int:
    """已发出的 LLM 调用次数（可观测性）。"""
    return _RL_CALLS[0]


def available() -> bool:
    """网关是否可用。带缓存：可用结果永久缓存，不可用只缓存 30 秒。

    负缓存必须过期，否则一次探测超时会毒化整个进程（见 _NEG_TTL 的注释）。
    探测失败时要把已配置的通道逐条试完：只探主通道就判死，会让「备用通道」
    形同虚设——写作端看到 False 直接走模板骨架，压根没机会触发 chat_text
    里的自动故障转移，等于换个 API 也救不回来。
    """
    global _CACHE_AVAIL, _CACHE_AT
    now = time.time()
    if _CACHE_AVAIL is True:
        return True
    if _CACHE_AVAIL is False and (now - _CACHE_AT) < _NEG_TTL:
        return False

    n = len(channels())
    for i in range(max(1, n)):
        try:
            c = _client_or_none()
            if c is None:
                break
            client, model, extra = c
            # 轻握手：极小 prompt 验证连通性与鉴权（带限流感知重试）
            # 快速失败：探测只是决定「走 LM 还是走确定性兜底」，不值得长时间等待
            _raw_chat(client, model, extra,
                      [{"role": "user", "content": "ping"}], max_tokens=2,
                      timeout=6, max_attempts=1)
            _CACHE_AVAIL = True
            _CACHE_AT = now
            return True
        except Exception:
            if i + 1 < n and switch_channel():
                continue
    _CACHE_AVAIL = False
    _CACHE_AT = now
    return False


def reset_cache() -> None:
    global _CACHE_AVAIL, _CACHE_AT
    _CACHE_AVAIL = None
    _CACHE_AT = 0.0


def _is_fatal(exc) -> bool:
    """鉴权/参数类错误：重试无意义，立即抛出。"""
    s = str(exc)
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", 0) in (401, 403):
        return True
    if any(k in s for k in ("401", "403", "Incorrect API key", "invalid api key",
                            "Authentication", "Unauthorized")):
        return True
    return False


def _is_retryable(exc) -> bool:
    """限流/过载/超时/连接类错误：可退避后重试。"""
    s = str(exc)
    if isinstance(exc, (RateLimitError, APIStatusError, APITimeoutError,
                         APIConnectionError, APIError)):
        # openai 的 429 / 5xx / 超时 / 连接错误都走退避重试
        sc = getattr(exc, "status_code", None)
        if sc in (401, 403):
            return False
        return True
    keys = ("429", "queue.full", "RateLimit", "rate limit", "Too Many Requests",
            "500", "502", "503", "504", "Internal Server", "Bad Gateway",
            "Service Unavailable", "Gateway Timeout", "timeout", "timed out",
            "Connection", "RemoteDisconnect", "reset by peer", "ECONNRESET")
    return any(k in s for k in keys)


def _throttle() -> None:
    """进程内节流：等待到「上次调用 + 最小间隔」之后，再放行一次调用。
    若熔断器已开路，立即抛错让调用方走确定性兜底，避免空转。"""
    if _cb_open():
        raise RuntimeError("LLM 网关熔断中（连续限流），转确定性兜底")
    if _RL_MAX_CALLS and _RL_CALLS[0] >= _RL_MAX_CALLS:
        raise RuntimeError(
            f"LLM 调用次数已达上限 {_RL_MAX_CALLS}，强制降级到确定性兜底")
    with _RL_LOCK:
        now = time.time()
        wait = _RL_MIN_INTERVAL - (now - _RL_LAST[0])
        if wait > 0:
            time.sleep(wait)
        _RL_LAST[0] = time.time()
        _RL_CALLS[0] += 1


def _raw_chat(client, model, extra, messages, *, max_tokens, timeout,
              temperature=0.3, max_attempts=4, backoff_base=3.0, cap=25.0):
    """底层调用 + 限流感知指数退避 + 熔断器。429/5xx/超时/连接错误会重试；鉴权错误立即抛。

    重试预算：最多 max_attempts 次（默认 4），退避 = backoff_base * 2**attempt + 抖动，
    单步上限 cap（默认 25s）。单次最坏停留 ≈ 3+6+12+25 ≈ 46s；连续限流达阈值由熔断器开路，
    避免整条流水线空熬。成功则复位熔断器；可重试失败则累计熔断计数。
    """
    last = None
    for attempt in range(max_attempts):
        _throttle()  # 全局节流（含最小间隔、计数上限、熔断开路检查）
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, extra_body=extra,
                temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            )
            _cb_trip_or_reset(False)  # 成功 → 复位熔断
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_fatal(e):
                raise
            if not _is_retryable(e):
                raise
            _cb_trip_or_reset(True)  # 可重试失败 → 累计熔断计数
            # 主通道连续限流/过载时换备用通道立刻重试，不在同一条上空熬退避。
            # 换通道不换产出标准：仍然是同一个模型调用链，绝不因此降级交付。
            if (attempt < max_attempts - 1 and _FAILOVER_AFTER
                    and _CB_FAILS[0] >= _FAILOVER_AFTER
                    and len(channels()) > 1 and switch_channel()):
                c = _client_or_none()
                if c:
                    client, model, extra = c
                    print(f"[LLM] 备用通道接管第 {attempt + 2} 次调用…")
                    continue
            if attempt >= max_attempts - 1:
                break
            delay = min(backoff_base * (2 ** attempt), cap) + random.uniform(0, 1.5)
            print(f"[LLM] 网关限流/过载，第 {attempt + 1} 次重试，{delay:.1f}s 后继续…")
            time.sleep(delay)
    raise RuntimeError(f"LLM 调用失败（已重试 {max_attempts} 次）：{last!r}")


def chat_text(prompt: str, system: str | None = None,
              temperature: float = 0.3, max_tokens: int = 1500,
              timeout: int = 90) -> str:
    """调用网关返回纯文本。失败抛异常，由调用方决定兜底。"""
    c = _client_or_none()
    if c is None:
        raise RuntimeError("LLM 网关不可用（缺 Key 或依赖）")
    client, model, extra = c
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = _raw_chat(client, model, extra, messages,
                  max_tokens=max_tokens, timeout=timeout,
                  temperature=temperature, max_attempts=3, backoff_base=2.5, cap=20.0)
    return (r.choices[0].message.content or "").strip()


def chat_json(prompt: str, system: str | None = None,
              temperature: float = 0.3, max_tokens: int = 2500,
              timeout: int = 120) -> dict:
    """调用网关并尽力解析 JSON。解析失败返回 {}（调用方兜底）。"""
    text = chat_text(prompt, system=system, temperature=temperature,
                     max_tokens=max_tokens, timeout=timeout)
    if not text:
        return {}
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    blob = m.group(1) if m else text
    s, e = blob.find("{"), blob.rfind("}")
    if s >= 0 and e > s:
        blob = blob[s:e + 1]
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# 全局系统提示：所有 Agent 默认遵循的「不编造、基于素材」铁律。
FACT_FIRST_SYSTEM = (
    "你是严谨的科研助手。铁律：只能依据用户提供的真实素材（文献标题/摘要、实验数字、"
    "方法名）做判断与生成；不得编造任何素材之外的文献、数字、方法或事实。"
    "输出严格遵循用户要求的 JSON 结构，不要输出 JSON 以外的任何内容。"
)
