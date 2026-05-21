import math
import os
import unittest

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEEPSEEK_R1_DISTILL_QWEN_1_5B

# JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust-remote-code --dist-init-addr=0.0.0.0:10011 --nnodes=1 --tp-size=1 --device=tpu --random-seed=27 --node-rank=0 --mem-fraction-static=0.8 --chunked-prefill-size=8192 --download-dir=/tmp --dtype=bfloat16 --precompile-bs-paddings 1 64 --max-running-requests 64 --max-total-tokens 257536 --skip-server-warmup --attention-backend=fa --precompile-token-paddings 8192 --page-size=64 --disable-overlap-schedule --log-requests --log-requests-level=3 --enable-precision-tracer --use-sort-for-toppk-minp

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"


print("Running on Google TPU")
# Default engine configuration
DEFAULT_ENGINE_CONFIG = {
    "model_path": DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    "random_seed": 27,
    "device": "tpu",
    "chunked_prefill_size": 8192,
    "dtype": "bfloat16",
    "max_running_requests": 64,
    "page_size": 64,
    "max_total_tokens": 257536,
    "precompile_token_paddings": [8192],
    "precompile_bs_paddings": [1, 64],
    "use_sort_for_toppk_minp": True,
    "mem_fraction_static": 0.8,
    "disable_overlap_schedule": True,
    "trust_remote_code": True,
    "skip_server_warmup": True,
    "tp_size": 1,
    "enable_precision_tracer": True,
    "log_level": "info",
}


class TestLogprobsDense(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang-Jax Engine with {DEEPSEEK_R1_DISTILL_QWEN_1_5B}...")
        cls.engine = Engine(**DEFAULT_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - shutdown the engine."""
        cls.engine.shutdown()

    def assert_scalar_logprobs(self, logprobs, key):
        for i, (logprob, _, _) in enumerate(logprobs):
            if logprob is None:
                continue
            self.assertFalse(
                isinstance(logprob, (list, tuple)),
                f"{key}[{i}] should be a scalar logprob, got {type(logprob)}",
            )
            shape = getattr(logprob, "shape", ())
            self.assertEqual(shape, (), f"{key}[{i}] should be scalar, got shape {shape}")

    def test_logprobs(self):
        ## prompt = "please introduce yourself"
        input_ids = [151646, 151644, 30021, 19131, 6133, 151645, 151648, 198]

        sampling_params = {"n": 1, "top_k": 1, "max_new_tokens": 3}
        start_len = 1
        top_logprobs_num = 2
        token_ids_logprob = [10]

        output = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
        output_meta = output["meta_info"]
        ## number check
        self.assertEqual(
            len(output_meta["input_token_logprobs"]),
            len(input_ids) - start_len,
            "input_token_logprobs is invalid",
        )
        self.assert_scalar_logprobs(output_meta["input_token_logprobs"], "input_token_logprobs")
        self.assertEqual(
            len(output_meta["output_token_logprobs"]),
            len(output["output_ids"]),
            "output_token_logprobs is invalid",
        )
        self.assertEqual(
            len(output_meta["input_top_logprobs"]),
            len(input_ids) - start_len,
            "intput_top_logprobs is invalid",
        )
        self.assertEqual(
            len(output_meta["output_top_logprobs"]),
            len(output["output_ids"]),
            "output_top_logprobs is invalid",
        )

        for i, (output_top_logprob, output_id) in enumerate(
            zip(output_meta["output_top_logprobs"], output["output_ids"])
        ):
            self.assertEqual(
                len(output_top_logprob),
                top_logprobs_num,
                f"output_top_logprobs at {i} is invalid",
            )
            self.assertEqual(
                output_top_logprob[0][1],
                output_id,
                "output id is is not the top logprob",
            )
            max_logprobs = output_top_logprob[0][0]
            for j, logprob in enumerate(output_top_logprob):
                self.assertGreaterEqual(max_logprobs, logprob[0], "the logprob is not the max")

        self.assertEqual(
            len(output_meta["input_token_ids_logprobs"]),
            len(input_ids) - start_len,
            "input_token_ids_logprobs is invalid",
        )
        self.assertEqual(
            len(output_meta["output_token_ids_logprobs"]),
            len(output["output_ids"]),
            "output_token_ids_logprobs is invalid",
        )

        expected_output_logprobs = [
            [-0.87109375, 32313, "Okay"],
            [0.0, 11, ","],
            [-0.318359375, 773, " so"],
        ]
        self.check_output(output_meta, "output_token_logprobs", expected_output_logprobs)

        # use another expected, because jax compiler fused ops will introduce numerical precision issue
        expected_output_logprobs = [
            [-0.921875, 32313, "Okay"],
            [0.0, 11, ","],
            [-0.3515625, 773, " so"],
        ]
        output = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
        )
        output_meta = output["meta_info"]
        self.assertEqual(output_meta["cache_miss_count"], 0, "occur cache_miss")
        self.check_output(output_meta, "output_token_logprobs", expected_output_logprobs)

        sampling_params = {"n": 1, "temperature": 0.6, "top_p": 0.95, "max_new_tokens": 3}

        output = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
        output_meta = output["meta_info"]
        # With temperature>0 sampling, exact tokens depend on RNG state.
        # Only verify structural correctness here.
        self.assertEqual(
            len(output_meta["output_token_logprobs"]),
            3,
            "output_token_logprobs length mismatch",
        )
        for i, logprob in enumerate(output_meta["output_token_logprobs"]):
            self.assertLessEqual(logprob[0], 0.0, f"logprob at {i} should be non-positive")

        output = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
        )
        output_meta = output["meta_info"]
        self.assertEqual(output_meta["cache_miss_count"], 0, "occur cache_miss")
        self.assertEqual(
            len(output_meta["output_token_logprobs"]),
            3,
            "output_token_logprobs length mismatch",
        )

    def check_output(self, actual, key, expected):
        for i, logprob in enumerate(actual[key]):
            self.assertEqual(logprob[0], expected[i][0], f"{logprob[0]} logprob is invalid")
            self.assertEqual(logprob[1], expected[i][1], f"{logprob[1]} output id is invalid")
            self.assertEqual(logprob[2], expected[i][2], f"{logprob[2]} token is invalid")


DP_REGRESSION_ENGINE_CONFIG = {
    **DEFAULT_ENGINE_CONFIG,
    "tp_size": 4,
    "dp_size": 2,
    "enable_precision_tracer": False,
    "chunked_prefill_size": 128,
    "max_total_tokens": 32768,
    "precompile_token_paddings": [128, 256, 512, 1024],
    "precompile_bs_paddings": [1, 4, 8],
}


class TestLogprobsDpChunkedPrefill(unittest.TestCase):
    """Regression for the dp>1 chunked-prefill skip-tracking bug.

    Pre-fix, `process_batch_result_prefill` used `skip_stream_req: Req | None`,
    a single slot. On dp>1, each dp rank can have its own chunked-in-flight req,
    so all but the last-assigned one leaked into `stream_output` with
    `input_token_logprobs_val == None`. TokenizerManager then either crashed
    (`'NoneType' object is not iterable`) or, if coerced to `[]`, returned
    truncated logprobs that produced inf PPL downstream.

    This test forces multiple in-flight chunked reqs across dp ranks by
    submitting prompts longer than chunked_prefill_size, with dp_size=2, and
    asserts every req returns full, finite, scalar input_token_logprobs.
    """

    @classmethod
    def setUpClass(cls):
        print(f"Launching dp=2 tp=2 Engine with {DEEPSEEK_R1_DISTILL_QWEN_1_5B}...")
        cls.engine = Engine(**DP_REGRESSION_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_dp2_multi_req_chunked_prefill_logprobs(self):
        # 4 prompts of varying lengths, all > chunked_prefill_size=128 so each
        # spans multiple chunks. Different lengths ensure chunk boundaries
        # don't coincide, maximizing the chance multiple reqs are mid-prefill
        # in the same batch step (the dp>1 leak condition).
        base = [151646, 151644, 30021, 19131, 6133, 151645, 151648, 198]
        prompts = [
            base * 20,  # 160 tokens
            base * 25,  # 200 tokens
            base * 30,  # 240 tokens
            base * 35,  # 280 tokens
        ]
        sampling_params = [{"n": 1, "top_k": 1, "max_new_tokens": 2}] * len(prompts)

        output = self.engine.generate(
            input_ids=prompts,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=[0] * len(prompts),
        )

        self.assertEqual(len(output), len(prompts), "must return one result per req")

        for i, (out, prompt) in enumerate(zip(output, prompts)):
            meta = out["meta_info"]
            self.assertIsNotNone(
                meta.get("input_token_logprobs"),
                f"req[{i}]: input_token_logprobs is None — chunked-skip leak",
            )
            self.assertEqual(
                len(meta["input_token_logprobs"]),
                len(prompt),
                f"req[{i}]: expected {len(prompt)} input logprobs, got "
                f"{len(meta['input_token_logprobs'])} — truncated by chunked-skip leak",
            )
            for j, (logprob, token_id, _) in enumerate(meta["input_token_logprobs"]):
                # With logprob_start_len=0, the first token has no preceding
                # context to score against — it MUST be None. Every other
                # position MUST be a finite scalar; partial chunked-skip leaks
                # show up as scattered None/inf in the middle of the sequence,
                # which a length-only check would miss.
                if j == 0:
                    self.assertIsNone(
                        logprob,
                        f"req[{i}][0]: expected None (no prior context), got {logprob}",
                    )
                    self.assertEqual(
                        token_id,
                        prompt[j],
                        f"req[{i}][0]: token_id mismatch {token_id} vs prompt {prompt[j]}",
                    )
                    continue
                self.assertIsNotNone(
                    logprob,
                    f"req[{i}][{j}]: logprob is None mid-sequence — chunked-skip leak",
                )
                shape = getattr(logprob, "shape", ())
                self.assertEqual(
                    shape,
                    (),
                    f"req[{i}][{j}]: logprob must be scalar, got shape {shape}",
                )
                self.assertFalse(
                    isinstance(logprob, (list, tuple)),
                    f"req[{i}][{j}]: logprob must be scalar, got {type(logprob)}",
                )
                self.assertTrue(
                    math.isfinite(float(logprob)),
                    f"req[{i}][{j}]: logprob is non-finite ({logprob}) — "
                    f"likely empty-logprobs→inf-PPL from chunked-skip leak",
                )
                self.assertEqual(
                    token_id,
                    prompt[j],
                    f"req[{i}][{j}]: token_id mismatch {token_id} vs prompt "
                    f"{prompt[j]} — index alignment regression",
                )


if __name__ == "__main__":
    unittest.main()
