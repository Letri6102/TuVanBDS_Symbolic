# %% [markdown]
# # Train PhoBERT Semantic Index trên Google Colab GPU
#
# Bật GPU trước khi chạy:
# Runtime -> Change runtime type -> Hardware accelerator -> GPU.
#
# Notebook này train semantic index bằng `vinai/phobert-base-v2`.
# Đây là embedding/index training, không phải fine-tune PhoBERT bằng label.

# %%
!nvidia-smi

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Mount Google Drive
#
# Cách khuyến nghị:
# - Upload cả folder project `bds_neurosymbolic_project` lên Google Drive.
# - Đảm bảo trong project có thư mục `data/raw` chứa các file Excel crawl.
# - Sửa `PROJECT_DIR` đúng với vị trí folder trong Drive.

# %%
from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = "/content/drive/MyDrive/bds_neurosymbolic_project"
%cd "$PROJECT_DIR"

# %% [markdown]
# ## 2. Cài dependency
#
# Colab thường đã có sẵn PyTorch GPU. Không nên upgrade torch nếu không cần,
# vì có thể làm lệch CUDA runtime của Colab.

# %%
!pip -q install transformers sentencepiece underthesea openpyxl pandas numpy scikit-learn plotly streamlit openai

# %% [markdown]
# ## 3. Cấu hình PhoBERT chạy GPU
#
# Nếu bị CUDA out-of-memory, giảm `PHOBERT_BATCH_SIZE` xuống `8` hoặc `4`.

# %%
import os

os.environ["SEMANTIC_BACKEND"] = "phobert"
os.environ["PHOBERT_MODEL_NAME"] = "vinai/phobert-base-v2"
os.environ["PHOBERT_DEVICE"] = "cuda"
os.environ["PHOBERT_BATCH_SIZE"] = "16"
os.environ["PHOBERT_MAX_LENGTH"] = "256"

# %% [markdown]
# ## 4A. Chạy full pipeline
#
# Dùng cell này nếu muốn đọc lại Excel, chuẩn hoá, xoá trùng, sinh symbolic facts,
# KG triples, rồi train semantic index PhoBERT.

# %%
!python scripts/run_pipeline.py

# %% [markdown]
# ## 4B. Chỉ train lại PhoBERT semantic index
#
# Dùng cell này nếu `outputs/bds_clean_symbolic.csv` đã có sẵn và bạn chỉ muốn
# train lại `outputs/semantic_index.json`.
#
# Nếu đã chạy 4A thì không cần chạy 4B.

# %%
# !python scripts/train_semantic_index.py

# %% [markdown]
# ## 5. Kiểm tra metadata
#
# Kết quả đúng sẽ có:
# - `backend`: `phobert`
# - `model_name`: `vinai/phobert-base-v2`
# - `embedding_dim`: thường là `768`

# %%
!cat outputs/semantic_index_metadata.json

# %% [markdown]
# ## 6. Test query recommendation

# %%
import ast
import pandas as pd

from Recommendation.recommender import recommend
from Recommendation.advisor import build_advisory_response, build_customer_friendly_advice
from Recommendation.llm_agent import build_agent_report
from Recommendation.semantic import load_semantic_index


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    list_cols = [
        "amenity_tags",
        "symbolic_facts",
        "risk_flags",
        "triggered_rules",
        "data_flags",
        "parse_warnings",
        "rule_evidence",
    ]
    for col in list_cols:
        if col in df.columns:
            def parse(x):
                if isinstance(x, list):
                    return x
                if pd.isna(x):
                    return []
                try:
                    parsed = ast.literal_eval(str(x))
                    return parsed if isinstance(parsed, list) else []
                except Exception:
                    return []
            df[col] = df[col].apply(parse)
    return df


df = parse_list_columns(pd.read_csv("outputs/bds_clean_symbolic.csv"))
semantic_index = load_semantic_index("outputs/semantic_index.json")

query = "Tài chính 3 tỷ thì nên mua nhà như thế nào?"
profile, recs = recommend(query, df, top_k=5, relax=True, semantic_index=semantic_index)

print(profile)
recs[
    [
        "property_id",
        "district",
        "property_type",
        "price_billion",
        "area_m2",
        "bedrooms",
        "bathrooms",
        "floors",
        "legal_class",
        "final_score",
        "symbolic_score",
        "semantic_score",
        "semantic_backend",
    ]
]

# %% [markdown]
# ## 6B. Nhập query trực tiếp và sinh bài phân tích Top 3
#
# Mặc định cell này dùng fallback narrative có cấu trúc. Nếu muốn gọi LLM API thật:
#
# ```python
# import os
# os.environ["LLM_PROVIDER"] = "openai"
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["LLM_MODEL"] = "model-ban-muon-dung"
# ```

# %%
from IPython.display import display, Markdown

query = input("Nhập nhu cầu BĐS: ")
top_k_raw = input("Top K muốn xem, mặc định 5: ").strip()
top_k = int(top_k_raw) if top_k_raw else 5

profile, recs = recommend(
    query,
    df,
    top_k=top_k,
    relax=True,
    semantic_index=semantic_index,
)

display(Markdown("## Gợi ý cho khách hàng"))
display(Markdown(build_customer_friendly_advice(query, profile, recs).replace("\n", "\n\n")))

display(Markdown("## Tư vấn kỹ thuật tổng quan"))
display(Markdown(build_advisory_response(query, profile, recs, df).replace("\n", "\n\n")))

display(Markdown("## Bài phân tích Top 3"))
display(Markdown(build_agent_report(query, profile, recs, use_llm=True).replace("\n", "\n\n")))

display(Markdown("## Bảng kiểm chứng"))
show_cols = [
    "property_id",
    "district",
    "property_type",
    "price_billion",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "floors",
    "legal_class",
    "final_score",
    "symbolic_score",
    "semantic_score",
    "semantic_backend",
    "amenity_tags",
    "symbolic_facts",
    "risk_flags",
]
display(recs[[c for c in show_cols if c in recs.columns]])

# %% [markdown]
# ## 6C. Cấu hình ChatGPT API và test gọi LLM
#
# Cell này nhập API key bằng `getpass`, không in key ra notebook.
# Không lưu API key cứng vào file project hoặc Google Drive.
#
# Nếu model mặc định không có trong tài khoản của bạn, đổi `LLM_MODEL`
# sang model bạn đang được cấp quyền trong OpenAI dashboard.

# %%
import os
from getpass import getpass
from openai import APIStatusError, AuthenticationError, OpenAI, RateLimitError

USE_CHATGPT_API = False
api_key = getpass("Nhập OpenAI API key, bỏ trống nếu muốn dùng fallback narrative: ").strip()

if not api_key:
    os.environ["LLM_PROVIDER"] = "none"
    print("Chưa nhập API key. Hệ thống sẽ dùng fallback narrative, không gọi ChatGPT API.")
else:
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = input("Model ChatGPT, mặc định gpt-4o-mini: ").strip() or "gpt-4o-mini"
    os.environ["LLM_TEMPERATURE"] = "0.25"

    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        test_response = client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=[
                {"role": "system", "content": "Bạn là trợ lý kiểm thử API. Trả lời ngắn gọn bằng tiếng Việt."},
                {"role": "user", "content": "API đã hoạt động chưa?"},
            ],
            temperature=0.1,
        )
        USE_CHATGPT_API = True
        print(test_response.choices[0].message.content)
    except AuthenticationError:
        os.environ["LLM_PROVIDER"] = "none"
        print("API key không hợp lệ. Hệ thống sẽ dùng fallback narrative.")
    except RateLimitError as exc:
        os.environ["LLM_PROVIDER"] = "none"
        print("OpenAI API đang hết quota/rate limit. Hệ thống sẽ dùng fallback narrative.")
        print("Cách xử lý: kiểm tra Billing/Usage trong OpenAI dashboard, nạp credit hoặc dùng key khác.")
    except APIStatusError as exc:
        os.environ["LLM_PROVIDER"] = "none"
        print(f"OpenAI API trả lỗi {exc.status_code}. Hệ thống sẽ dùng fallback narrative.")


# %% [markdown]
# ## 6D. Test query BĐS với ChatGPT API thật
#
# Cell này dùng:
# - Recommender neuro-symbolic để chọn Top-K.
# - PhoBERT semantic score từ `semantic_index.json`.
# - ChatGPT API để viết bài phân tích Top 3 bằng ngôn ngữ tự nhiên.

# %%
query = input("Nhập nhu cầu BĐS: ")
top_k_raw = input("Top K muốn xem, mặc định 5: ").strip()
top_k = int(top_k_raw) if top_k_raw else 5

profile, recs = recommend(
    query,
    df,
    top_k=top_k,
    relax=True,
    semantic_index=semantic_index,
)

llm_report = build_agent_report(query, profile, recs, use_llm=globals().get("USE_CHATGPT_API", False))

display(Markdown("## Gợi ý cho khách hàng"))
display(Markdown(build_customer_friendly_advice(query, profile, recs).replace("\n", "\n\n")))

display(Markdown("## Bài phân tích từ ChatGPT Agent"))
display(Markdown(llm_report.replace("\n", "\n\n")))

display(Markdown("## Bảng kiểm chứng Top-K"))
show_cols = [
    "property_id",
    "district",
    "property_type",
    "price_billion",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "floors",
    "legal_class",
    "final_score",
    "symbolic_score",
    "semantic_score",
    "semantic_backend",
    "amenity_tags",
    "symbolic_facts",
    "risk_flags",
]
display(recs[[c for c in show_cols if c in recs.columns]])

# %% [markdown]
# ## 7. Download output cần đem về máy local

# %%
from google.colab import files

files.download("outputs/semantic_index.json")
files.download("outputs/semantic_index_metadata.json")
files.download("outputs/bds_clean_symbolic.xlsx")
