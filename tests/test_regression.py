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
from paper.lit_material import (  # noqa: E402
    gist_block, gist_items, normalize_gist, topic_overlap,
)
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
from verify_multitype import (  # noqa: E402
    _backup_before_build, _gateway_ok, _needs_revise,
    _restore_after_failed_build,
)
from engine import selection_ledger as SL  # noqa: E402
from engine import lit_plan as LP  # noqa: E402
from agents import paper_review_agent as PRA  # noqa: E402


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


class TestBuildGuard(unittest.TestCase):
    """网关一抖就把成稿换成兜底骨架，是实测造成过数据损失的缺陷。"""

    def test_backup_restores_full_set(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "out"
            (p / "06_paper").mkdir(parents=True)
            names = ("paper_full.md", "paper_full.docx", "WRITE_STATUS.json",
                     "07_eval_verdict.json")
            for n in names:
                (p / "06_paper" / n).write_text("GOOD:" + n, encoding="utf-8")
            saved = _backup_before_build(p)
            self.assertEqual(len(saved), 4)
            for n in names:
                (p / "06_paper" / n).write_text("FALLBACK", encoding="utf-8")
            _restore_after_failed_build(p, saved)
            for n in names:
                self.assertEqual((p / "06_paper" / n).read_text(encoding="utf-8"),
                                 "GOOD:" + n)
                self.assertFalse((p / "06_paper" / (n + ".prebuild")).exists())

    def test_gateway_probe_rechecks_cache(self):
        with patch.object(LLM, "available", return_value=False) as av, \
                patch.object(LLM, "reset_cache") as rc:
            self.assertFalse(_gateway_ok())
        rc.assert_called_once()
        av.assert_called_once()

    def test_build_timeout_never_hangs(self):
        """构建子进程必须有整体超时，否则一次退避重试能挂住整个验证。"""
        import subprocess
        import verify_multitype as VM
        with patch.object(VM.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired("generate_paper.py", 1)):
            rc, tail = VM._build(Path("unused"), timeout=1)
        self.assertEqual(rc, 124)
        self.assertIn("超时", tail)


class TestGistMaterial(unittest.TestCase):
    """gist 是写作端与评审端共用的第二份真实素材，两端必须同源。"""

    def test_placeholder_gists_are_dropped(self):
        for bad in ("未知", "无", "N/A", "unknown", "-", None, 5, "   "):
            self.assertEqual(normalize_gist(bad), "", f"应拒绝 {bad!r}")
        self.assertEqual(normalize_gist("  用KNN  预测  风险 "), "用KNN 预测 风险")

    def test_items_keep_numbering_and_skip_placeholders(self):
        papers = [
            {"gist_cn": "第一篇用随机森林"},
            {"gist_cn": "未知"},
            {"gist_cn": "第三篇做综述"},
        ]
        items = gist_items(papers)
        self.assertEqual(items, ["[1] 第一篇用随机森林", "[3] 第三篇做综述"])

    def test_writer_and_judge_share_exact_items(self):
        papers = [{"gist_cn": f"第{i}篇做了某研究"} for i in range(1, 6)]
        papers[2]["gist_cn"] = "暂无"
        w, ws = gist_block(papers, audience="writer")
        j, js = gist_block(papers, audience="judge")
        items = gist_items(papers)
        self.assertTrue(all(x in w for x in items))
        self.assertTrue(all(x in j for x in items))
        self.assertEqual(ws["n_used"], js["n_used"])
        self.assertIn("概要里没写的结论", w)
        self.assertIn("属于", j)

    def test_gist_enters_fact_pack_and_judge_gt(self):
        from pathlib import Path
        from agents.eval_agent import _gist_gt
        papers = [{"gist_cn": "测试八周体育干预对农村中学儿童心血管风险的影响"}]
        facts, st = build_fact_pack("case_study", {"title": "测试题目"},
                                    [{"num": 1}], "[1] 清单", {"papers": papers})
        self.assertEqual(st["gists_used"], 1)
        self.assertIn("测试八周体育干预", facts)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "01_literature.json").write_text(
                json.dumps({"papers": papers}, ensure_ascii=False), encoding="utf-8")
            self.assertIn("[1] 测试八周体育干预", _gist_gt(p))

    def test_topic_overlap_flags_mismatched_pool(self):
        good = [{"title": "Machine learning for cardiovascular disease risk prediction"},
                {"title": "Deep learning detects atrial fibrillation from wearable ECG"}]
        bad = [{"title": "After-school program quality in rural middle schools"},
               {"title": "Teacher motivation and student satisfaction"}]
        q = "machine learning cardiovascular disease risk prediction"
        self.assertGreater(topic_overlap(q, good), 0.5)
        self.assertLess(topic_overlap(q, bad), 0.25)

    def test_cite_tail_drops_placeholder(self):
        from paper.generate_paper import _cite_tail
        self.assertEqual(_cite_tail({"gist": "未知", "usable_in": "方法"}), "")
        self.assertEqual(_cite_tail({"gist": "用KNN预测", "usable_in": ""}), "：用KNN预测")

    def test_gist_reaches_actual_writer_prompt(self):
        """不依赖网关：捕获 _write_batches 真正发给模型的提示词。"""
        from paper import generate_paper as GP
        papers = [{"gist_cn": "该研究测试八周体育干预的可行性"}]
        facts, st = build_fact_pack("case_study", {"title": "县域中学课后服务"},
                                    [{"num": 1}], "[1] 清单", {"papers": papers})
        seen = []

        def fake(prompt, **kw):
            seen.append(prompt)
            return {"abstract": "占位"}

        with patch.object(GP.LLM, "chat_json", side_effect=fake):
            GP._write_batches("case_study", facts)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all("[1] 该研究测试八周体育干预的可行性" in p for p in seen))
        self.assertTrue(all("概要里没写的结论" in p for p in seen))

    def test_neutral_citation_block_no_fake_experiment_claim(self):
        from paper.generate_paper import _role_citation_block
        refs = [{"num": 1, "title": "某研究", "venue": "期刊", "year": 2024,
                 "role": "方法", "tier": "core", "gist": "做了某研究"}]
        neutral = _role_citation_block(refs, neutral=True)
        self.assertNotIn("统一实验协议下重新评估", neutral)
        self.assertIn("不据概要推断文献未写明的结论", neutral)
        plain = _role_citation_block(refs, neutral=False)
        self.assertIn("统一实验协议下重新评估", plain)


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


class TestSelectionLedger(unittest.TestCase):
    """选材台账 + 人工偏好：单条意见必须立刻生效，且不影响引用编号顺序。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._p = patch.object(SL, "PREF_PATH", Path(self.tmp.name) / "prefs.json")
        self._p.start()
        self.papers = [
            {"id": "a", "title": "Deep learning for heart risk", "year": 2024,
             "venue": "Nature", "abstract": "clinical cohort"},
            {"id": "b", "title": "A preprint on diabetes", "year": 2018,
             "venue": "arXiv", "abstract": "preprint"},
            {"id": "c", "title": "Survey of cardiovascular models", "year": 2022,
             "venue": "IEEE", "abstract": "review"},
        ]

    def tearDown(self):
        self._p.stop()
        self.tmp.cleanup()

    def test_avoid_rule_removes_and_keeps_order(self):
        SL.add_rule("avoid", "keyword", "preprint", note="别用预印本")
        kept, applied = SL.apply_preferences(self.papers, "心血管")
        self.assertEqual([p["id"] for p in kept], ["a", "c"])
        self.assertEqual(applied["n_before"], 3)
        self.assertEqual(applied["n_after"], 2)

    def test_year_rule_and_boost(self):
        SL.add_rule("avoid", "year", "<=2020")
        SL.add_rule("boost", "keyword", "survey", weight=0.2)
        kept, _ = SL.apply_preferences(self.papers)
        self.assertEqual([p["id"] for p in kept], ["a", "c"])
        self.assertGreater(kept[1].get("pref_bonus", 0), 0)

    def test_keep_overrides_drop(self):
        SL.add_decision("A preprint on diabetes", "drop", note="不相关")
        SL.add_decision("A preprint on diabetes", "keep", note="导师要求保留")
        kept, _ = SL.apply_preferences(self.papers)
        self.assertEqual(len(kept), 3)
        self.assertGreater(kept[1].get("pref_bonus", 0), 0)

    def test_record_selection_covers_all_candidates(self):
        pool = self.papers + [{"id": "d", "title": "Dropped one", "year": 2019}]
        chosen = self.papers[:2]
        d = Path(self.tmp.name) / "out"
        info = SL.record_selection("心血管", pool, chosen, d, selector="model")
        led = json.loads((d / SL.LEDGER_NAME).read_text(encoding="utf-8"))
        self.assertEqual(info["n_candidates"], 4)
        self.assertEqual(info["n_selected"], 2)
        self.assertEqual(len(led["decisions"]), 4)
        self.assertEqual(sum(1 for x in led["decisions"] if x["selected"]), 2)
        self.assertTrue(all(x["by"] in ("model", "rule", "human") for x in led["decisions"]))
        self.assertTrue(SL.attach_review(d, {"weak_ids": ["d"],
                                             "missing_themes": ["缺综述"]}))
        led2 = json.loads((d / SL.LEDGER_NAME).read_text(encoding="utf-8"))
        self.assertEqual(led2["review"]["missing_themes"], ["缺综述"])

    def test_preferences_and_notes_reach_writer_fact_pack(self):
        """闭环的消费端：人的表态必须真的出现在写作提示里，不能只存在文件里。"""
        SL.add_rule("avoid", "keyword", "preprint", note="别用预印本")
        SL.add_review_note("out_x", "P01", "accepted", "照改", "第2章补样本说明")
        facts, _ = build_fact_pack("survey", {"title": "心血管综述"},
                                   [{"num": 1}], "[1] 清单", {"papers": []})
        self.assertIn("用户偏好", facts)
        self.assertIn("人工批注", facts)
        self.assertIn("第2章补样本说明", facts)

    def test_preference_prompt_and_review_notes(self):
        SL.add_rule("avoid", "keyword", "preprint", note="别用预印本")
        SL.add_review_note("out_x", "P01", "accepted", "照改", "第2章补样本说明")
        SL.add_review_note("out_x", "P02", "rejected", "", "别再提框架图")
        pp = SL.preference_prompt("心血管")
        self.assertIn("不要选", pp)
        self.assertIn("别用预印本", pp)
        notes = [x["detail"] for x in SL.review_notes_for()]
        self.assertTrue(any("人工确认要改" in x for x in notes))
        self.assertTrue(any("已被人工驳回" in x for x in notes))
        self.assertIn("【用户偏好", SL.preference_prompt())


class TestPaperReview(unittest.TestCase):
    """文本诊断：不打分、定位句必须可核实、兜底不许装作读过全文。"""

    def _md(self) -> str:
        return ("## 封面\n姓名：某某\n\n## 诚信声明\n本人声明。\n\n"
                "## 摘要\n本文研究某问题。\n\n"
                "## 第1章 绪论\n" + "背景与意义。" * 60 + "\n\n"
                "## 第2章 方法\n" + "方法描述。" * 60 + "\n\n"
                "## Abstract\nThis is an English abstract.\n\n"
                "## 参考文献\n[1] A.\n")

    def test_fallback_skips_front_matter_and_english(self):
        r = PRA._fallback(self._md())
        self.assertFalse(r["llm_available"])
        self.assertIn("未完成文本诊断", r["verdict"])
        joined = " ".join(r["cross_cutting"])
        for bad in ("封面", "诚信声明", "Abstract", "摘要"):
            self.assertNotIn(bad, joined, "前置材料/英文摘要不该被判成展开不足")
        self.assertIn("未完成逐章诊断", PRA._render(r))
        self.assertNotIn("分", PRA._render(r).split("## 整体判断")[0])

    def test_fabricated_quotes_are_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "paper.md").write_text(self._md(), encoding="utf-8")
            fake = {"verdict": "修改后可提交", "overall": "整体判断",
                    "strengths": [],
                    "chapters": [{"chapter": "第1章 绪论", "judgment": "尚可",
                                  "problems": [
                                      {"quote": "背景与意义。", "why": "太泛",
                                       "how": "补具体数据", "severity": "中"},
                                      {"quote": "文中根本没有这句话", "why": "编的",
                                       "how": "删掉", "severity": "高"}]}],
                    "cross_cutting": [], "top_actions": []}
            with patch.object(PRA.LLM, "available", return_value=True), \
                    patch.object(PRA.LLM, "chat_json", return_value=fake):
                r = PRA.run(str(p / "paper.md"), str(p))
            self.assertEqual(r["dropped_unverifiable_quotes"], 1)
            self.assertEqual(len(r["chapters"][0]["problems"]), 1)
            self.assertEqual(r["chapters"][0]["problems"][0]["id"], "P01")
            self.assertTrue((p / "08_paper_review.md").exists())
            self.assertTrue((p / "08_paper_review.json").exists())

    def test_run_falls_back_when_gateway_down(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "paper.md").write_text(self._md(), encoding="utf-8")
            with patch.object(PRA.LLM, "available", return_value=False):
                r = PRA.run(str(p / "paper.md"), str(p))
        self.assertFalse(r["llm_available"])
        self.assertEqual(r["provenance"], "rule")


class TestFeedbackLoop(unittest.TestCase):
    """「评审反馈 → 检索计划 → 下轮抓取」这条自我改进闭环必须真的连起来。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._p = patch.object(SL, "PREF_PATH", Path(self.tmp.name) / "prefs.json")
        self._p.start()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self._p.stop()
        self.tmp.cleanup()

    def test_feedback_query_consumed_once(self):
        n = SL.add_feedback_queries(["wearable ECG atrial fibrillation review",
                                      "wearable ECG atrial fibrillation review"],
                                     source="paper_review", note="缺可穿戴综述")
        self.assertEqual(n, 1)
        first = SL.take_feedback_queries()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["source"], "paper_review")
        self.assertEqual(SL.take_feedback_queries(), [], "同一条反馈不该每轮重复并入")

    def test_plan_marks_feedback_origin_and_rounds(self):
        fb = [{"query": "ECG wearable survey", "source": "paper_review", "note": "缺综述"}]
        plan = LP.create("心电监测", ["machine learning ECG", "ECG wearable survey"],
                         ["crossref", "openalex"], lm_driven=True, feedback=fb)
        origins = {g["query"]: g["origin"] for g in plan["strategy"]["query_groups"]}
        self.assertEqual(origins["ECG wearable survey"], "feedback")
        self.assertEqual(origins["machine learning ECG"], "planner")
        self.assertEqual(plan["strategy"]["purpose_source"], "keyword_heuristic")
        LP.add_round(plan, "refetch", ["ECG benchmark dataset"], 7, "审评缺口补抓")
        LP.finish(plan, 120, 40, ["缺基准数据集"])
        path = LP.write(self.root, plan)
        loaded = LP.load(self.root)
        self.assertEqual(len(loaded["rounds"]), 2)
        self.assertEqual(loaded["outcome"]["n_selected"], 40)
        self.assertTrue(Path(path).exists())

    def test_review_saves_suggested_queries_only_when_llm_read(self):
        md = "## 第1章 绪论\n" + "正文内容。" * 80 + "\n"
        paper = self.root / "paper.md"
        paper.write_text(md, encoding="utf-8")
        fake = {"verdict": "修改后可提交", "overall": "文献覆盖偏窄",
                "strengths": [], "chapters": [], "cross_cutting": [],
                "top_actions": [], "suggested_queries": ["cardiac wearable ECG survey"]}
        with patch.object(PRA.LLM, "available", return_value=True), \
                patch.object(PRA.LLM, "chat_json", return_value=fake):
            r = PRA.run(str(paper), str(self.root))
        self.assertEqual(r["feedback_queries_saved"], 1)
        self.assertEqual(SL.take_feedback_queries()[0]["source"], "paper_review")
        with patch.object(PRA.LLM, "available", return_value=False):
            r2 = PRA.run(str(paper), str(self.root))
        self.assertEqual(r2["feedback_queries_saved"], 0, "规则兜底不配给检索建议")

    def test_review_report_shows_applied_feedback(self):
        run_dir = self.root / "run"
        out = run_dir / "06_paper"
        out.mkdir(parents=True)
        (out / "paper.md").write_text("## 第1章 绪论\n" + "正文。" * 80, encoding="utf-8")
        plan = LP.create("心电", ["ECG wearable survey"], ["crossref"],
                         feedback=[{"query": "ECG wearable survey",
                                    "source": "literature_review", "note": "上轮缺综述"}])
        LP.write(run_dir, plan)
        with patch.object(PRA.LLM, "available", return_value=False):
            r = PRA.run(str(out / "paper.md"), str(out))
        self.assertEqual(len(r["applied_feedback"]), 1)
        self.assertIn("本次检索已应用的历史反馈", (out / "08_paper_review.md").read_text(encoding="utf-8"))

    def test_human_choices_mark_ledger(self):
        run_dir = self.root / "run2"
        papers = [{"id": "a", "title": "Paper A"}, {"id": "b", "title": "Paper B"}]
        SL.record_selection("测试", papers, papers, run_dir, selector="model")
        res = SL.apply_human_choices(run_dir, ["a"], ["b"], "人工勾选：B 与题目无关")
        self.assertEqual(res["ledger_updated"], 2)
        led = json.loads((run_dir / SL.LEDGER_NAME).read_text(encoding="utf-8"))
        by_num = {r["id"]: r for r in led["decisions"]}
        self.assertEqual(by_num["a"]["by"], "human")
        self.assertFalse(by_num["b"]["selected"])
        self.assertEqual(by_num["b"]["by"], "human")
        kept, applied = SL.apply_preferences(papers, "测试")
        self.assertEqual([p["id"] for p in kept], ["a"])
        self.assertEqual(applied["removed"][0]["by"], "human")
