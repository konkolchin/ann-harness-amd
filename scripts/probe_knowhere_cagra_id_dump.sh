#!/usr/bin/env bash
# Instrument Knowhere Catch2 to dump GPU_CUVS_CAGRA neighbor IDs for triage.
#
# Correct failure map after 0052 (NN_DESCENT) - line numbers +1 vs stock:
#   :206  Search With Bitset   REQUIRE(recall > 0.7)     <- CAGRA 0.0
#   :243  Search TopK          IVF_PQ near-miss only
#   :329  Search Simple Bitset REQUIRE(recall >= 0.8)    <- CAGRA 0.0
# Serialize has no hard recall assert (only CHECK ids[i]==i). Cosine/Hamming green.
#
# This probe dumps:
#   - bitset / simple-bitset result vs GT IDs (the red paths)
#   - pre-serialize vs post-deserialize search IDs (confirm serialize is OK)
#
# Usage on amd-rx7900xtx:
#   export WORKDIR=~/rocmds_check_gfx1100
#   cd ~/ann-harness-amd && git pull --ff-only
#   bash scripts/probe_knowhere_cagra_id_dump.sh
#   SKIP_REBUILD=1 bash scripts/probe_knowhere_cagra_id_dump.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
KH="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
TG="${KH}/tests/ut/test_gpu_search.cc"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${KH}/build:${WORKDIR}/install/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"

if [ ! -f "${TG}" ]; then
  echo "ERROR: ${TG} missing" >&2
  exit 1
fi

echo "==> instrument ${TG}"
python3 - "${TG}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "CAGRA_ID_DUMP_V1"

if marker in text:
    print(f"already instrumented ({marker})")
    sys.exit(0)

if "#include <iostream>" not in text:
    text = text.replace(
        "#include <string>\n",
        "#include <string>\n#include <iostream>\n#include <sstream>\n",
        1,
    )

helper = f'''
// {marker} begin
static void
DumpCagraIds(const std::string& name, const char* tag, const knowhere::DataSet& res,
             const knowhere::DataSet* gt = nullptr) {{
    if (name.find("CAGRA") == std::string::npos) {{
        return;
    }}
    const auto nq = res.GetRows();
    const auto k = res.GetDim();
    const auto* ids = res.GetIds();
    const auto* dist = res.GetDistance();
    int64_t neg = 0, zero = 0;
    for (int64_t i = 0; i < nq * k; ++i) {{
        if (ids[i] < 0) {{
            ++neg;
        }}
        if (ids[i] == 0) {{
            ++zero;
        }}
    }}
    std::ostringstream oss;
    oss << "[{marker}] tag=" << tag << " name=" << name << " nq=" << nq
        << " k=" << k << " neg1=" << neg << " id0_count=" << zero << " sample=";
    const int64_t show_q = std::min<int64_t>(nq, 6);
    for (int64_t i = 0; i < show_q; ++i) {{
        oss << " q" << i << "=[";
        for (int64_t j = 0; j < k; ++j) {{
            oss << ids[i * k + j];
            if (dist != nullptr) {{
                oss << ":" << dist[i * k + j];
            }}
            if (j + 1 < k) {{
                oss << ",";
            }}
        }}
        oss << "]";
    }}
    if (gt != nullptr) {{
        const auto gk = gt->GetDim();
        const auto* gids = gt->GetIds();
        oss << " gt=";
        const int64_t show_g = std::min<int64_t>(nq, 4);
        for (int64_t i = 0; i < show_g; ++i) {{
            oss << " q" << i << "=[";
            for (int64_t j = 0; j < std::min(gk, k); ++j) {{
                oss << gids[i * gk + j];
                if (j + 1 < std::min(gk, k)) {{
                    oss << ",";
                }}
            }}
            oss << "]";
        }}
    }}
    std::cerr << oss.str() << std::endl;
}}
// {marker} end

'''

needle = "#ifdef KNOWHERE_WITH_CUVS\n"
if needle not in text:
    raise SystemExit("cannot find KNOWHERE_WITH_CUVS guard")
text = text.replace(needle, needle + helper, 1)

old_bs = """                float recall = GetKNNRecall(*gt.value(), *results.value());
                if (percentage == 0.98f) {
                    REQUIRE(recall > 0.4f);
                } else {
                    REQUIRE(recall > 0.7f);
                }"""
new_bs = """                float recall = GetKNNRecall(*gt.value(), *results.value());
                DumpCagraIds(name, "bitset", *results.value(), gt.value().get());
                CAPTURE(recall, percentage);
                if (percentage == 0.98f) {
                    REQUIRE(recall > 0.4f);
                } else {
                    REQUIRE(recall > 0.7f);
                }"""
if old_bs not in text:
    raise SystemExit("bitset dump anchor not found - lab source differs")
text = text.replace(old_bs, new_bs, 1)

old_topk = """            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= std::get<1>(topKTuple));"""
new_topk = """            float recall = GetKNNRecall(*gt.value(), *results.value());
            DumpCagraIds(name, "topk", *results.value(), gt.value().get());
            CAPTURE(recall);
            REQUIRE(recall >= std::get<1>(topKTuple));"""
if old_topk not in text:
    raise SystemExit("topk dump anchor not found")
text = text.replace(old_topk, new_topk, 1)

old_ser = """        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        idx_.Deserialize(bs);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();"""
new_ser = """        {
            auto pre = idx.Search(query_ds, json, nullptr);
            REQUIRE(pre.has_value());
            DumpCagraIds(name, "serialize_pre", *pre.value(), nullptr);
        }
        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        idx_.Deserialize(bs);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        DumpCagraIds(name, "serialize_post", *results.value(), nullptr);
        auto ids = results.value()->GetIds();"""
if old_ser not in text:
    raise SystemExit("serialize dump anchor not found")
text = text.replace(old_ser, new_ser, 1)

old_simple = """        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.8f);
    }

    SECTION("Test Gpu Index Search Cosine Metric")"""
new_simple = """        float recall = GetKNNRecall(*gt.value(), *results.value());
        DumpCagraIds(name, "simple_bitset", *results.value(), gt.value().get());
        CAPTURE(recall);
        REQUIRE(recall >= 0.8f);
    }

    SECTION("Test Gpu Index Search Cosine Metric")"""
if old_simple not in text:
    raise SystemExit("simple_bitset dump anchor not found")
text = text.replace(old_simple, new_simple, 1)

path.write_text(text, encoding="utf-8")
print(f"instrumented {path}")
PY

grep -n 'CAGRA_ID_DUMP\|DumpCagraIds\|serialize_pre\|simple_bitset' "${TG}" | head -40

if [ "${SKIP_REBUILD:-0}" != "1" ]; then
  echo "==> rebuild knowhere_tests"
  export ROCM_PATH
  export PATH="${ROCM_PATH}/llvm/bin:${PATH}"
  cd "${KH}"
  if [ ! -d build ]; then
    echo "ERROR: ${KH}/build missing" >&2
    exit 1
  fi
  cmake --build build -j"$(nproc)" --target knowhere_tests 2>&1 | tail -50
fi

KT=""
for c in "${KH}/build/tests/ut/knowhere_tests" "${KH}/build/knowhere_tests"; do
  if [ -x "$c" ]; then KT="$c"; break; fi
done
if [ -z "${KT}" ]; then
  echo "ERROR: knowhere_tests not found" >&2
  exit 1
fi

LOG="${LOG_DIR}/cagra_id_dump_${TS}.log"
echo "==> Catch2 Test All GPU Index -s -> ${LOG}"
set +e
"${KT}" "Test All GPU Index" -s 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
set -e
echo "exit=${rc}"

echo ""
echo "==> CAGRA_ID_DUMP lines"
grep -n 'CAGRA_ID_DUMP_V1' "${LOG}" | tee "${LOG_DIR}/cagra_id_dump_${TS}_extract.txt" || echo "(none - instrumentation/build mismatch?)"

echo ""
echo "==> FAILED blocks"
grep -n -A6 'FAILED:' "${LOG}" | head -80 || true

echo ""
echo "DONE: ${LOG}"
echo "Extract: ${LOG_DIR}/cagra_id_dump_${TS}_extract.txt"
echo ""
cat <<'EOF'
How to read dumps:
  tag=bitset / simple_bitset  -> red paths; check neg1 (all -1?) vs wrong ids vs GT
  tag=topk                    -> should look sane if TopK CAGRA is green
  tag=serialize_pre vs _post  -> IDs should match; if both wrong, search bug; if only post, serialize bug
EOF
