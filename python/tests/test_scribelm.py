import unittest
from scribelm import scribe, critique_text, rewrite_text, write_post, generate_social


class TestScribeLM(unittest.TestCase):
    def test_critique_authenticity(self) -> None:
        hem = (
            "Skills vs MCP? Wrong question.\n\n"
            "Both are tools for building AI agents. But they solve different problems:\n"
            "- Skills = what your agent knows how to do\n"
            "- MCP = what your agent can access\n\n"
            "One teaches. One connects. You need both."
        )
        corp = (
            "In today's rapidly evolving landscape, organizations must comprehensively delve "
            "into multifaceted paradigms and leverage holistic synergy to facilitate suboptimal workflows."
        )
        pos = critique_text(hem)
        neg = critique_text(corp)
        self.assertEqual(pos["verdict"], "authentic_hemanth_voice")
        self.assertGreater(pos["p_hemanth"], 0.70)
        self.assertEqual(neg["verdict"], "needs_hemanth_rewrite")
        self.assertLess(neg["p_hemanth"], 0.35)

    def test_rewrite_nullspan(self) -> None:
        draft = (
            "Baked nullspan a single-pass structured extraction with exact [start, end] character spans, "
            "calibrated p_null abstention, algebraic invariant checks (subtotal == nodes * hourly_rate)."
        )
        w = scribe(draft)
        res = w.rewrite()
        self.assertIn("Meet nullspan", res["text"])
        self.assertGreater(res["after_p_hemanth"], res["before_p_hemanth"])

    def test_write_and_social(self) -> None:
        w = scribe("Zero-Overhead Token Budgets", category="python")
        post = w.write()
        self.assertIn("# Zero-Overhead Token Budgets", post["text"])
        soc = w.social(post["text"])
        self.assertIn("TL;DR:", soc["reddit"])
        self.assertIn("Title:", soc["hn"])


if __name__ == "__main__":
    unittest.main()
