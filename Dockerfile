# AutoResearch —— 通用 Auto Research 引擎镜像
# 基础镜像固定 Python 3.11（与本地运行环境一致）
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# 系统依赖：中文字体（matplotlib 图表中文正常显示）+ libgomp1（xgboost/lightgbm 运行期依赖）
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先装纯 Python 依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 可选进阶后端：--build-arg EXTRAS=1 时一并安装 xgboost / lightgbm / torch
# 默认不装，保持镜像精简；装了也不影响主链路（引擎按需动态 import）
# 注意：torch 走 PyTorch 官方 CPU 索引，xgboost/lightgbm 走 PyPI，必须分开装，
#      否则 --index-url 作用于全部包会导致 xgboost/lightgbm 找不到而构建失败。
ARG EXTRAS=0
RUN if [ "$EXTRAS" = "1" ]; then \
        pip install --no-cache-dir xgboost lightgbm; \
        pip install --no-cache-dir torch --index-url https://download.pytorch.com/whl/cpu; \
    fi

# 复制工程源码（.dockerignore 已排除 .env / 本地产物）
COPY . .

# 挂载目标目录（挂载时被宿主目录覆盖；不挂载时也能正常落盘）
RUN mkdir -p /app/out_docker /app/my_specs

# 非 root 用户运行，符合分发安全基线
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 入口：默认跑内置示例 spec；可用 docker run ... run --spec ... --out ... 覆盖
ENTRYPOINT ["python", "auto_research.py"]
CMD ["run", "--spec", "engine/tasks/proof/wine_sklearn_spec.json", "--out", "/app/out_docker"]
