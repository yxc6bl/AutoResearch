"""AutoResearch 确定性回归测试（不联网、不调 LLM）。

    python -m unittest discover -s tests -v

这些用例固化的是「一旦悄悄改动就会造成实质损失」的不变量：
零编造边界、写作端与评审端事实基础同源、重写意见分发、闸门判据不循环、
素材完备度、网关负缓存可恢复。运行时间应在数秒内，不依赖任何外部服务。
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.lit_abstracts import abstract_block, clean_abstract, load_papers  # noqa: E402
from paper.paper_type_specs import TYPE_SPECS, build_fact_pack  # noqa: E402
from paper.spec_check import check_spec  # noqa: E402
from paper.generate_paper import (  # noqa: E402
    _locate_claims, _review_block, _review_digest, _sanitize_citations,
)
from agents.eval_agent import (  # noqa: E402
    _abstract_gt, _gist_gt, _ground_truth_for, _points_at_text,
)
from engine import gates as G  # noqa: E402
from engine import llm_client as LLM  # noqa: E402
from paper.thesis_template import chapters_for  # noqa: E402
from verify_multitype import _needs_revise  # noqa: E402


def _papers(n: int = 5) -> list[dict]:
    out = []
    for i in range(1, n + 1):
        out.append({
            "title": f"Study number {i} on topic",
            "authors": f"Author {i}",
            "year": 2020 + i,
            "venue": "Journal of Tests",
            "abstract": (f"This study investigates problem {i} using method {i} "
                         f"on a cohort of patients and reports a measurable improvement "
                         f"in risk stratification compared with baseline approaches. "
                         f"The conclusion is that the approach generalises to similar settings."),
        })
    return out


class TestAbstractCleaning(unittest.TestCase):
    def test_placeholders_and_short_text_rejected(self):
        for bad in ("No abstract available", "N/A", "暂无摘要", "短", None, 123,
                    "<jats:p>Hi</jats:p>"):
            self.assertEqual(clean_abstract(bad), "", f"应拒绝 {bad!r}")

    def test_tags_and_whitespace(self):
        s = clean_abstract("<jats:p>We propose   a novel\n method for risk prediction "
                           "using routinely collected clinical variables.</jats:p>")
        self.assertNotIn("<jats:p>", s)
        self.assertNotIn("\n", s)
        self.assertIn("We propose a novel method", s)

    def test_truncation_marks_ellipsis(self):
        s = clean_abstract("A" * 900, limit=120)
        self.assertLessEqual(len(s), 125)
        self.assertTrue(s.endswith("…"))

    def test_non_string_input_safe(self):
        self.assertEqual(abstract_block([None, "x", 5, {"title": "t"}])[0], "")


class TestAbstractBlock(unittest.TestCase):
    def test_numbering_stays_aligned_and_ascending(self):
        papers = _papers(8)
        papers[0]["abstract"] = ""       # [1] 无摘要
        papers[3]["abstract"] = "N/A"   # [4] 占位符
        block, st = abstract_block(papers, per_paper=200, max_chars=8000)
        self.assertEqual(st["n_total"], 8)
        self.assertEqual(st["n_with_abstract"], 6)
        nums = [int(x) for x in re.findall(r"^\[(\d+)\]", block, re.M)]
        self.assertEqual(nums, sorted(nums))
        self.assertNotIn(1, nums)
        self.assertNotIn(4, nums)
        self.assertIn(2, nums)

    def test_limits_respected(self):
        block, st = abstract_block(_papers(40), per_paper=200,
                                   max_papers=7, max_chars=100000)
        self.assertLessEqual(st["n_used"], 7)
        self.assertLessEqual(block.count("\n["), 7)

    def test_char_budget(self):
        _, st = abstract_block(_papers(60), per_paper=300,
                               max_papers=60, max_chars=3000)
        self.assertLessEqual(st["chars"], 3000 + 400)

    def test_no_abstract_falls_back_to_empty(self):
        papers = [{**p, "abstract": ""} for p in _papers(4)]
        block, st = abstract_block(papers)
        self.assertEqual(block, "")
        self.assertEqual(st["n_used"], 0)


class TestFactPackGrounding(unittest.TestCase):
    """最关键的契约：写作端看到的摘要 == 评审官核对的摘要（逐字）。

    摘要接入默认关闭（见 TestAbstractsDisabledByDefault），这里显式开启以验证
    启用路径本身是同源的。
    """

    def setUp(self):
        self._env = patch.dict(os.environ, {"AR_LIT_ABSTRACTS": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def test_writer_and_judge_share_exact_same_block(self):
        papers = _papers(6)
        lit = {"papers": papers}
        for pt in [k for k in TYPE_SPECS]:
            facts, st = build_fact_pack(pt, {"title": "测试题目"},
                                        [{"num": i} for i in range(1, 7)],
                                        "[1] 清单", lit)
            ab, _ = abstract_block(papers)
            self.assertIn(ab, facts, f"{pt} 事实包缺摘要块")
            self.assertGreater(st["n_used"], 0)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "01_literature.json").write_text(
                json.dumps(lit, ensure_ascii=False), encoding="utf-8")
            ab, _ = abstract_block(papers)
            self.assertEqual(_abstract_gt(p), ab)
            self.assertEqual(load_papers(p), papers)

    def test_no_abstract_keeps_legacy_wording(self):
        lit = {"papers": [{**p, "abstract": ""} for p in _papers(3)]}
        facts, st = build_fact_pack("survey", {"title": "测试题目"},
                                    [{"num": i} for i in range(1, 4)], "[1] 清单", lit)
        self.assertEqual(st["n_used"], 0)
        self.assertIn("没有摘要", facts)
        self.assertNotIn("文献摘要节选（真实数据", facts)

    def test_with_abstract_switches_constraint(self):
        lit = {"papers": _papers(3)}
        facts, _ = build_fact_pack("survey", {"title": "测试题目"},
                                   [{"num": i} for i in range(1, 4)], "[1] 清单", lit)
        self.assertIn("部分文献的真实摘要节选", facts)
        self.assertIn("该文献报告", facts)
        self.assertIn("全文严禁任何具体数字", facts)

    def test_abstract_numbers_are_stripped(self):
        """数值脱敏是这轮最关键的安全阀：模型看不到数字，就不会给文献补数字。"""
        papers = _papers(2)
        papers[0]["abstract"] = (
            "The study included 700,217 participants in three national biobanks "
            "and reported 92.3% accuracy with a 1D-CNN model in 2021.")
        block, st = abstract_block(papers)
        self.assertGreater(st["n_used"], 0)
        self.assertIn("〔数值略〕", block)
        self.assertNotIn("700,217", block)
        self.assertNotIn("92.3", block)
        self.assertIn("three national biobanks", block)
        self.assertIn("1D-CNN", block)  # 方法名里的数字不该被毁掉

    def test_judge_gt_contains_abstracts_for_all_non_experimental(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "01_literature.json").write_text(
                json.dumps({"papers": _papers(6)}, ensure_ascii=False), encoding="utf-8")
            (p / "task_spec.json").write_text(json.dumps({
                "paper_type": "system_design", "title": "t",
                "system_spec": {"tech_stack": ["Python"]}}, ensure_ascii=False),
                encoding="utf-8")
            ab, _ = abstract_block(_papers(6))
            for pt in ["survey", "system_design", "case_study",
                       "technical_report", "research_proposal"]:
                info = _ground_truth_for(pt, p, {}, None)
                self.assertIn(ab, info["gt"], f"{pt} 评审事实基础缺摘要")
                self.assertTrue(info["abstracts_grounded"])
            mc = _ground_truth_for("method_comparison", p, {}, None)
            self.assertFalse(mc["abstracts_grounded"])


class TestAbstractsDisabledByDefault(unittest.TestCase):
    """四轮实测未能超过基线，因此默认走已验证的零幻觉路径。"""

    def test_writer_keeps_legacy_wording(self):
        with patch.dict(os.environ, {"AR_LIT_ABSTRACTS": ""}):
            facts, st = build_fact_pack("survey", {"title": "测试题目"},
                                        [{"num": i} for i in range(1, 4)],
                                        "[1] 清单", {"papers": _papers(3)})
        self.assertEqual(st["n_used"], 0)
        self.assertIn("没有摘要", facts)
        self.assertNotIn("文献摘要节选（真实数据", facts)

    def test_judge_gt_has_no_abstract_block(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.dict(os.environ, {"AR_LIT_ABSTRACTS": ""}):
            p = Path(td)
            (p / "01_literature.json").write_text(
                json.dumps({"papers": _papers(3)}, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(_abstract_gt(p), "")

    def test_gist_grounding_covers_citation_block_numbers(self):
        """引用块里的 gist 数字是真实元数据，必须进事实基础，否则必被误判编造。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            papers = _papers(3)
            papers[0]["gist_cn"] = "用KNN预测缺血性心血管病，数据来自Kaggle的918条记录。"
            (p / "01_literature.json").write_text(
                json.dumps({"papers": papers}, ensure_ascii=False), encoding="utf-8")
            gt = _gist_gt(p)
        self.assertIn("[1] 用KNN", gt)
        self.assertIn("918条记录", gt)
        self.assertIn("真实素材", gt)


class TestReviewRouting(unittest.TestCase):
    def test_digest_dedupes(self):
        d = _review_digest({"hallucinations": ["A", "A", "B"],
                            "unsupported_claims": ["B", "C"],
                            "suggestions": ["S"]})
        self.assertEqual(d["claims"], ["A", "B", "C"])
        self.assertEqual(d["suggestions"], ["S"])

    def test_claim_located_elsewhere_not_forwarded(self):
        review = {"hallucinations": ["第4章称每小时一次"],
                  "_locate": {"第4章称每小时一次": ["implementation"]}}
        self.assertEqual(_review_block(review, ["results"]), "")

    def test_suggestions_only_still_produces_block(self):
        """曾经的真实缺陷：只有建议时返回空串，重写轮等于原地重掷。"""
        b = _review_block({"suggestions": ["补一句口径说明"]}, ["implementation"])
        self.assertIn("补一句口径说明", b)
        self.assertIn("采纳边界", b)

    def test_unlocated_claims_broadcast_everywhere(self):
        review = {"hallucinations": ["某处数字没有依据"]}
        for fields in (["intro"], ["implementation"], ["results"]):
            self.assertIn("某处数字没有依据", _review_block(review, fields))

    def test_locate_claims_finds_chapter(self):
        from paper.thesis_template import chapters_for as cf
        md = ("## 第1章 绪论\n背景说明。\n\n"
              "## 第4章 实施过程\nAirflow 调度每小时一次，失败重试三次。\n\n"
              "## 第5章 结果与验证\n按验收标准核验。\n")
        review = {"hallucinations": ["第4章称“Airflow 调度每小时一次”，素材没有"]}
        loc = _locate_claims(review, md, cf("technical_report"))
        self.assertIn("第4章称“Airflow 调度每小时一次”，素材没有", loc)
        self.assertIn("implementation", loc["第4章称“Airflow 调度每小时一次”，素材没有"])


class TestDeterministicGuards(unittest.TestCase):
    def test_sanitize_out_of_range_citations(self):
        text = "见 [2] 与 [31] 至 [40]，另有 [99]，但 [1] 有效。"
        clean, n = _sanitize_citations(text, 5)
        self.assertGreater(n, 0)
        self.assertIn("[2]", clean)
        self.assertIn("[1]", clean)
        self.assertNotIn("[99]", clean)
        self.assertNotIn("[31]", clean)

    def test_points_at_text_respects_ground_truth(self):
        md = "正文称 AUC 达 99.3%，引用该文献。"
        gt = "该文献报告 AUC 达 99.3%。"
        self.assertFalse(_points_at_text(md, "论文称 AUC 达 99.3%，但事实清单无此数字", gt))
        self.assertTrue(_points_at_text(md, "论文称 AUC 达 99.3%，但事实清单无此数字", ""))

    def test_transcription_wording_is_not_hallucination(self):
        """评审官自己说「数字引用正确 / 未编造」，不得进零编造硬门。"""
        md = "文献[16]在三个国家生物银行的700217名参与者中结合代谢组学数据。"
        gt = "[16] 摘要：700,217 participants in three national biobanks"
        cases = [
            "数字引用正确，但论文未写明「该文献报告」，直接陈述为事实，属于转述不规范。",
            "该数字为文献[4]引用的WHO报告数据，属于转述不严谨，但未编造数字。",
        ]
        for c in cases:
            self.assertFalse(_points_at_text(md, c, gt), c[:40])

    def test_misattribution_survives_gt_number_present(self):
        """张冠李戴是真编造：数字虽然在 gt 里，但属于另一篇文献。"""
        md = "Nadakinamani等[9]利用包含11个临床特征的数据集设计模型。"
        gt = "[44] 摘要：a dataset that contains 11 features"
        c = ("文献[9]摘要中未提及「11个特征」，该数字可能来自文献[44]的摘要，"
             "此处将文献[44]的数据集特征数安到文献[9]头上，属于张冠李戴。")
        self.assertTrue(_points_at_text(md, c, gt))

    def test_revise_trigger_requires_hard_hallucination(self):
        no_hall = {"llm_judge": {"available": True, "hallucinations": [],
                                 "verdict": "revise"}}
        hall = {"llm_judge": {"available": True, "hallucinations": ["x"],
                              "verdict": "pass"}}
        self.assertFalse(_needs_revise(no_hall))
        self.assertTrue(_needs_revise(no_hall, all_signals=True))
        self.assertTrue(_needs_revise(hall))
        self.assertFalse(_needs_revise({"llm_judge": {"available": False}}))


class TestJudgeJSONRetry(unittest.TestCase):
    def test_bad_json_retries_with_tighter_output(self):
        """网关偶发坏 JSON 不该把评审官直接判成缺席（否则 G04 未验证）。"""
        from unittest.mock import patch
        from agents.eval_agent import llm_judge
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text("# 测试论文\n正文内容", encoding="utf-8")
            good = {"overall": 80, "scores": {}, "hallucinations": [], "issues": [],
                    "unsupported_claims": [], "suggestions": [], "verdict": "pass"}
            calls = []

            def fake(prompt, **kw):
                calls.append(kw.get("max_tokens"))
                return {} if len(calls) == 1 else good

            with patch.object(LLM, "available", return_value=True), \
                    patch.object(LLM, "chat_json", side_effect=fake):
                r = llm_judge(str(p), {}, 0, mode="general", gt="素材")
            self.assertTrue(r["available"])
            self.assertEqual(r["overall"], 80)
            self.assertEqual(len(calls), 2)
            self.assertGreater(calls[1], calls[0])

    def test_both_bad_json_reports_unavailable(self):
        from unittest.mock import patch
        from agents.eval_agent import llm_judge
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text("# 测试论文", encoding="utf-8")
            with patch.object(LLM, "available", return_value=True), \
                    patch.object(LLM, "chat_json", return_value={}):
                r = llm_judge(str(p), {}, 0, mode="general", gt="素材")
        self.assertFalse(r["available"])
        self.assertEqual(r["reason"], "bad_json")

    def test_abstract_rule_only_when_grounded(self):
        """默认关闭摘要时，不能把「正文严禁数字」的口径发给评审官。"""
        from agents.eval_agent import llm_judge
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "paper.md"
            p.write_text("# 测试论文", encoding="utf-8")
            good = {"overall": 80, "scores": {}, "hallucinations": [], "issues": [],
                    "unsupported_claims": [], "suggestions": [], "verdict": "pass"}
            seen = []

            def fake(prompt, **kw):
                seen.append(prompt)
                return good

            with patch.object(LLM, "available", return_value=True), \
                    patch.object(LLM, "chat_json", side_effect=fake):
                llm_judge(str(p), {}, 0, mode="general", gt="普通事实清单")
                llm_judge(str(p), {}, 0, mode="general", gt="摘要含〔数值略〕")
        self.assertNotIn("仅当事实清单附了摘要节选时适用", seen[0])
        self.assertIn("仅当事实清单附了摘要节选时适用", seen[1])


class TestSpecCheck(unittest.TestCase):
    def test_empty_spec_reports_gap(self):
        sc = check_spec("system_design", {"title": "校园二手书交易平台"})
        self.assertFalse(sc["ok"])
        self.assertLess(sc["score"], 30)
        self.assertIn("system_spec", [i["field"] for i in sc["items"]
                                      if i["status"] == "missing"])

    def test_complete_spec_is_100(self):
        sc = check_spec("technical_report", json.loads(
            (ROOT / "out_verify_technical_report" / "task_spec.json").read_text(encoding="utf-8")))
        self.assertTrue(sc["ok"])
        self.assertEqual(sc["score"], 100)

    def test_thin_fields_detected(self):
        sc = check_spec("research_proposal", {
            "title": "某研究开题报告",
            "proposal_spec": {"background": "背景很短", "questions": ["q"],
                              "content": ["c"], "methods": ["m"], "schedule": ["s"]}})
        self.assertTrue(any(i["status"] == "thin" for i in sc["items"]))


class TestGatesG03(unittest.TestCase):
    def _g03(self, write_status, delivery):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "06_paper").mkdir(parents=True)
            if write_status is not None:
                (p / "06_paper" / "WRITE_STATUS.json").write_text(
                    json.dumps(write_status), encoding="utf-8")
            if delivery is not None:
                (p / "06_paper" / "DELIVERY_STATUS.json").write_text(
                    json.dumps(delivery), encoding="utf-8")
            return [x for x in G.report(p, strict=True, write=False)["gates"]
                    if x["id"] == "G03"][0]["status"]

    def test_fallback_paper_still_fails(self):
        self.assertEqual(self._g03({"llm_written": False},
                                   {"deliverable": True, "llm_written": False}), "fail")

    def test_stale_not_deliverable_does_not_poison(self):
        """循环判据回归：一次不可交付之后，补齐评审必须能翻身。"""
        self.assertEqual(self._g03(
            {"llm_written": True, "llm_sections": ["a"] * 13, "chinese_chars": 10802},
            {"deliverable": False, "llm_written": True,
             "reasons": ["P0 闸门未通过/未验证：G04"]}), "pass")

    def test_missing_records_are_unknown(self):
        self.assertEqual(self._g03(None, {"deliverable": True}), "unknown")
        self.assertEqual(self._g03(None, None), "unknown")

    def test_legacy_delivery_record_works(self):
        self.assertEqual(self._g03(None, {"deliverable": True, "llm_written": True,
                                          "chinese_chars": 9000}), "pass")

    def test_sync_delivery_recovers(self):
        """交付状态必须能双向同步，不能只写不回。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "06_paper").mkdir(parents=True)
            dp = p / "06_paper" / "DELIVERY_STATUS.json"
            dp.write_text(json.dumps({
                "deliverable": False, "llm_written": True,
                "reasons": ["P0 闸门未通过/未验证：G04"],
                "hint": "不可交付"}, ensure_ascii=False), encoding="utf-8")
            rep = {"decision": {"deliverable": True, "basis": "全部通过",
                                "human_override": []},
                   "blocking": [], "unverified": []}
            G.sync_delivery(p, rep)
            d = json.loads(dp.read_text(encoding="utf-8"))
            self.assertTrue(d["deliverable"])
            self.assertFalse(any(str(r).startswith("P0 闸门") for r in d["reasons"]))


class TestLLMAvailabilityCache(unittest.TestCase):
    def test_negative_cache_expires(self):
        """负缓存必须会过期，否则一次 ping 失败毒化整个进程。"""
        orig_av, orig_at = LLM._CACHE_AVAIL, LLM._CACHE_AT
        orig_client = LLM._client_or_none
        try:
            LLM._CACHE_AVAIL = False
            LLM._CACHE_AT = time.time() - LLM._NEG_TTL - 1
            LLM._client_or_none = lambda: None  # 不联网
            self.assertFalse(LLM.available())
            self.assertGreaterEqual(LLM._CACHE_AT, time.time() - 2,
                                    "过期后应重新探测并刷新时间戳")
            self.assertFalse(LLM.available())  # TTL 内命中缓存，不再探测
        finally:
            LLM._CACHE_AVAIL, LLM._CACHE_AT = orig_av, orig_at
            LLM._client_or_none = orig_client

    def test_reset_cache_clears(self):
        orig_av, orig_at = LLM._CACHE_AVAIL, LLM._CACHE_AT
        try:
            LLM._CACHE_AVAIL = False
            LLM._CACHE_AT = time.time()
            LLM.reset_cache()
            self.assertIsNone(LLM._CACHE_AVAIL)
            self.assertEqual(LLM._CACHE_AT, 0.0)
        finally:
            LLM._CACHE_AVAIL, LLM._CACHE_AT = orig_av, orig_at


if __name__ == "__main__":
    unittest.main(verbosity=2)
