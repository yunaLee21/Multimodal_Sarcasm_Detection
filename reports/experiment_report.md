# Báo cáo thí nghiệm: CoT distillation và các hướng liên quan

File này được sinh tự động bởi `vimmsd.analysis.experiment_report`, không sửa tay. Nhận xét và quyết định nằm ở `reports/experiment_report_notes.yaml`. Mục chưa có kết quả được để trống. Số liệu trên val dùng để phát triển; private test chỉ xuất hiện sau bước đánh giá cuối.

## 1. Trạng thái các cổng quyết định

| Cổng | Điều kiện | Kết quả đo được | Quyết định |
|---|---|---|---|
| G0 | Phân cấp mềm không kém phẳng quá 1 điểm macro-F1 trên val | (a) kém 1.2 điểm: không đạt; (b) kém 0.5 điểm: đạt | Đi tiếp. Cột (a) kém 1,2 điểm, vượt ngưỡng 1 điểm, nên đã kiểm tra lại head phân cấp và loss: không thấy lỗi, test đều đạt. Theo seed, phân cấp hơn 0,2 điểm ở seed 42 và kém 2,7 và 1,0 điểm ở seed 1 và 2; độ lệch chuẩn giữa các seed của cột phẳng là 1,1 điểm, nên chênh lệch nằm trong mức nhiễu. |
| G1 | F1 `image-sarcasm` hoặc `multi-sarcasm` của teacher vượt mốc G0 trên val, parse lỗi dưới 5% | `val_v8_gemini-3.5-flash-lite`: đạt; `val_zero_shot_v6_gemini-3.5-flash-lite`: không đạt | Đạt về mặt hình thức với v8 nhưng sát ngưỡng, đi tiếp bằng thử distillation nhỏ thay vì sinh cả tập. `image-sarcasm` 0,097 so với mốc 0,077 (14/28 đúng, độ chính xác 5,3%); chỉ cần ít hơn 3 mẫu đúng là không đạt, và mốc này là trung bình của các seed (seed tốt nhất 0,130). `multi-sarcasm` 0,699, dưới mốc 0,853 (`distill_flat_b` có hiệu chỉnh prior; chưa hiệu chỉnh 0,836). Bước pilot-eval của kế hoạch chưa chạy. |
| G2 | κ ≥ 0,6 trên mọi tiêu chí; mỗi lớp khoảng 30 mẫu đạt rubric; nhóm 1 đạt rubric ≥ 80% |  |  |
| G3 | M1, M2 hoặc M3 hơn ô (b) tương ứng quá 1 độ lệch chuẩn |  |  |
| G4 | Báo cáo đủ các mục của giai đoạn 8 |  |  |
| G5 | E3 hơn cùng cấu hình không pretrain quá 1 độ lệch chuẩn trên val |  |  |

## 2. Thiết kế thí nghiệm

Chép nguyên từ `docs/experiment_design.md`.

<details>
<summary>1. Động cơ</summary>

Nhãn `multi-sarcasm` của ViMMSD không có nghĩa là "cả ảnh và text đều mỉa mai", mà là "châm biếm xuất hiện khi đọc ảnh cùng text". Một mẫu `multi` có thể thuộc 1 trong 4 trường hợp:

| # | Text đứng riêng | Ảnh đứng riêng | Khi ghép ảnh với text |
|---|---|---|---|
| 1 | mỉa mai | mỉa mai | mỉa mai |
| 2 | mỉa mai | không | mỉa mai |
| 3 | không | mỉa mai | mỉa mai |
| 4 | không | không | mỉa mai (chỉ xuất hiện khi ghép) |

Nhãn gốc không cho biết mẫu thuộc trường hợp nào. Trường hợp 2–4 đòi hỏi suy luận trên cả hai modality, và đây là phần các model fusion hiện tại (concat, cross-attention) dễ học nhầm thành shortcut từ text. Ý tưởng của hướng này: để một LLM đa phương thức mạnh (teacher) viết ra reasoning theo đúng khung 4 trường hợp, rồi chưng cất reasoning đó vào một model nhỏ (student, khoảng 300M tham số). Student là model chạy trên toàn bộ tập test.

Tài liệu này mô tả **3 phương pháp thí nghiệm**. Cả ba dùng chung dữ liệu reasoning (mục 3–4) và cùng cách đánh giá (mục 6), chỉ khác nhau ở đầu ra của student (mục 5):

- **M1, CoT distillation**: student là model 4 lớp phẳng, reasoning làm loss phụ.
- **M2, CoT distillation + hierarchical**: student có đầu ra phân cấp mềm. Tầng S hỏi "có châm biếm không", tầng C hỏi "loại châm biếm nào", và hai head T/I học phán đoán "text/ảnh đứng riêng" của teacher.
- **M3, CoT distillation + subclass**: lúc train, `multi` được tách thành 4 subclass theo 4 trường hợp ở bảng trên (dựa vào nhãn phụ của teacher), tổng cộng 7 lớp. Lúc suy luận, xác suất 4 subclass được cộng lại thành `multi`.

**M2 khác pipeline phân cấp của Vu et al. ở đâu.** Approach 3 của Vu et al. (2025) dùng 2 model nối tiếp với ngưỡng cứng: `multi` hay không, rồi 3 lớp còn lại. Cách này chỉ đạt F1 0,33 trên private test, kém nhất trong 3 hướng của bài, do lỗi của model đầu lan sang model sau. M2 vẫn là **một model, train end-to-end**. Các tầng được nối với nhau bằng phép nhân xác suất, không dùng ngưỡng cứng, nên gradient đi qua mọi tầng và không có lỗi lan truyền. Reasoning chỉ là tín hiệu giám sát lúc train ở cả ba phương pháp, không phải một tầng chạy lúc suy luận.

Tài liệu nền: STaR (Zelikman et al., 2022) cho cách lọc reasoning theo đáp án đúng và bước rationalization. Distilling step-by-step (Hsieh et al., 2023) cho cách dùng reasoning làm tác vụ phụ. MiDRE (Jana et al., 2025) cho cách dùng reasoning của LVLM trong MSD.

### 1.1. Bản đồ các hướng thí nghiệm

| Mã | Hướng | Vai trò | Cần teacher | Cần dữ liệu mới | Trạng thái | Mục |
|---|---|---|---|---|---|---|
| M1 | CoT distillation | phương pháp chính | có | không | có code, chưa chạy | 5.2 |
| M2 | CoT distillation + hierarchical | phương pháp chính | có ở bản đầy đủ, cột (a)/(b) thì không | không | có code, chưa chạy | 5.3 |
| M3 | CoT distillation + subclass | phương pháp chính | có | không | thiết kế, chưa có code | 5.4 |
| B1 | Phân cấp nối tiếp (tái lập Vu et al.) | baseline | không | không | có code, chưa chạy | 6.2 |
| B2 | 3 model riêng + stacking | baseline | có nếu T/I học nhãn teacher | không | thiết kế, chưa có code | 5.6 |
| B3 | Reasoning làm đầu vào | nhánh so sánh | có | không | chưa có config | 6.4 |
| E1 | kNN retrieval lúc suy luận | hậu xử lý, cộng thêm vào mọi model | không | không | thiết kế, chưa có code | 5.5 |
| E2 | Few-shot cho teacher | biến thể prompt | có | không | thiết kế, chưa có code | 3.4 |
| E3 | Pretrain trên dữ liệu sarcasm ngoài | mở rộng | chỉ khi dùng silver data | có | thiết kế, chưa có code | 5.7 |
| D1 | Kiểm tra shortcut | chẩn đoán | không | không | thiết kế, chưa có code | 6.6 |

Thứ tự chạy và cổng quyết định: `docs/experiment_plan.md`. Mọi lợi ích nêu ở các hướng trên đều là giả thuyết cho tới khi có số liệu.

</details>

<details>
<summary>2. Tổng quan pipeline</summary>

```
train (10,8k, có nhãn)
   │
   ▼
[A] Teacher sinh k reasoning/mẫu (khung 4 bước, không cho biết nhãn)
   │
   ▼
[B] Phân nhóm theo số lần đúng ──► sai cả k lần ──► [C] Rationalization (đưa nhãn đúng)
   │                                                        │
   ▼                                                        ▼
[D] Người kiểm tra theo rubric (toàn bộ nhóm 2, 3, mẫu ngẫu nhiên nhóm 1)
   │
   ▼
rationale_dataset.json (mẫu đạt rubric) ──► [E] Train student
                                               ├─ M1: 4 lớp phẳng + loss phụ
                                               ├─ M2: phân cấp mềm S → C, head T/I + loss phụ
                                               └─ M3: 7 lớp (multi tách 4 subclass), gộp lúc suy luận
                                                   │
                                                   ▼
                                     [F] Hiệu chỉnh prior, đánh giá trên private test
```

**Không chạy teacher trên val/test** để chọn hay lọc mẫu. Chỉ được chạy teacher trên val/test ở nhánh so sánh mục 6.4, nơi reasoning là đầu vào lúc suy luận và không dùng nhãn, và ở các phép đánh giá zero-shot trên val (giai đoạn 1 của kế hoạch), nơi kết quả không dùng để train hay lọc mẫu.

Các hướng ngoài pipeline này: E1 thêm vào sau bước [E], E2 thay đổi prompt ở bước [A], E3 thêm một bước pretrain trước [E], B2 là baseline độc lập, D1 chẩn đoán checkpoint đã train.

</details>

<details>
<summary>3. Sinh reasoning bằng teacher</summary>

### 3.1. Teacher

Nên thử trước trên 280 mẫu cân bằng (70 mỗi lớp) **lấy từ train**. Không lấy từ val được vì val chỉ có 1 mẫu `text-sarcasm`, và teacher chỉ được đánh giá chứ không học trên các mẫu này nên không gây rò rỉ. Sau đó chạy zero-shot trên toàn bộ val để so với baseline (chi tiết ở giai đoạn 1 của kế hoạch). Chọn teacher theo macro-F1 và F1 từng lớp:

- Qwen2.5-VL-7B-Instruct: miễn phí, chạy trên GPU server (bf16 cần khoảng 16 GB VRAM trở lên), hỗ trợ tiếng Việt khá tốt.
- Qwen2.5-VL-32B / 72B hoặc GPT-4o / Gemini qua API: mạnh hơn nhưng có chi phí. Ước tính chi phí trên 280 mẫu trước khi chạy cả tập.

Teacher phải vượt baseline hiện tại ít nhất ở hai lớp `image-sarcasm` và `multi-sarcasm`. Nếu không thì reasoning của nó không mang thêm thông tin, và nên sửa prompt hoặc đổi teacher trước khi đi tiếp.

### 3.2. Prompt (khung 4 bước)

Đầu vào: ảnh, caption, chữ OCR từ ảnh (lấy từ `image_preprocessed.json`). Prompt yêu cầu trả lời theo đúng thứ tự:

1. **Text đứng riêng**: chỉ đọc caption thì có mỉa mai không? Dẫn chứng từ caption.
2. **Ảnh đứng riêng**: chỉ nhìn ảnh, kể cả chữ trong ảnh, thì có mỉa mai không? Mô tả khách quan những gì thấy trong ảnh, không suy diễn.
3. **Khi ghép**: caption và ảnh có mâu thuẫn, phóng đại, hoặc tạo ra ý mỉa mai mới không?
4. **Kết luận**: một trong 4 nhãn, cùng hai nhãn phụ `text_sarcastic` và `image_sarcastic` (đúng/sai).

Định nghĩa 4 nhãn trong prompt phải chép đúng từ hướng dẫn gán nhãn của UIT DSC 2024. Cần đối chiếu lại định nghĩa trước khi chốt prompt.

Đầu ra yêu cầu ở dạng JSON để parse được:

```json
{
  "text_alone": {"sarcastic": false, "reason": "..."},
  "image_alone": {"sarcastic": true, "reason": "..."},
  "combined": {"reason": "..."},
  "label": "multi-sarcasm"
}
```

### 3.3. Self-consistency

Mỗi mẫu sinh k = 3–5 lần, với temperature khoảng 0,7. Ghi lại số lần teacher đoán đúng nhãn (`n_correct / k`). Tỉ lệ này vừa dùng để phân nhóm, vừa là thước đo độ khó của mẫu.

Chi phí: k = 3 trên 10,8k mẫu là khoảng 32k lần sinh. Chạy trên GPU server, mỗi GPU một shard, lưu cache để chạy tiếp được giống `scripts/extract_image_text.py` (`CODEBASE.md` mục 7.3).

### 3.4. Biến thể few-shot (E2)

Thay vì zero-shot, mỗi prompt kèm vài mẫu train giống nhất cùng nhãn của chúng. Mục tiêu là tăng độ chính xác của teacher và giảm số mẫu nhóm 3.

- **Tìm ví dụ:** dùng cosine trên vector PhoBERT và CLIP pretrained, không phụ thuộc checkpoint student.
- **Nguồn ví dụ:** chỉ lấy từ train. Luôn loại chính mẫu đang hỏi, và loại mọi mẫu prompt-dev, pilot-eval khỏi nguồn ví dụ.
- **Nội dung ví dụ:** bản đầu chỉ dùng caption, OCR, mô tả ảnh và nhãn, không chèn thêm ảnh để tiết kiệm VRAM.
- **Điều kiện dùng:** so với zero-shot trên prompt-dev, chỉ dùng khi F1 của `image-sarcasm` hoặc `multi-sarcasm` tăng. Mỗi lớp ở tập này chỉ có 40 mẫu, nên chênh lệch nhỏ chưa đủ để kết luận.

**Đánh đổi:** ví dụ gần trùng có thể làm teacher chép nhãn hàng xóm mà không thật sự suy luận. Khi đó `n_correct / k` không còn đo đúng độ khó của mẫu, nên phải ghi rõ biến thể đã dùng trong `prompt_version` và không trộn hai biến thể trong cùng một lần phân nhóm.

</details>

<details>
<summary>4. Lọc và kiểm tra</summary>

### 4.1. Phân nhóm

| Nhóm | Điều kiện | Xử lý | Người kiểm tra |
|---|---|---|---|
| 1 | đúng ≥ ⌈k/2⌉ lần | chọn 1 reasoning đúng | lấy ngẫu nhiên 10–20% |
| 2 | đúng 1 đến < ⌈k/2⌉ lần | chọn reasoning đúng tốt nhất | toàn bộ |
| 3 | sai cả k lần | chạy **rationalization**: đưa nhãn đúng, yêu cầu teacher giải thích theo khung 4 bước | toàn bộ, đánh dấu `rationalized=true` |

**Không chỉ giữ mẫu teacher đoán đúng.** Teacher thường đúng ở trường hợp 1 (dễ) và sai ở trường hợp 2–4 (khó). Bỏ nhóm 3 tức là bỏ đúng phần reasoning cần học nhất.

### 4.2. Rubric cho người kiểm tra

Mỗi tiêu chí chỉ chấm đạt hoặc không đạt. Một reasoning được giữ khi đạt cả 4 tiêu chí:

1. Đánh giá text đứng riêng đúng.
2. Mô tả ảnh đúng với ảnh, không bịa chi tiết (hallucination).
3. Lập luận khi ghép ảnh với text hợp lý, có dẫn chứng cụ thể từ ảnh hoặc caption.
4. Kết luận khớp với lập luận, và hai nhãn phụ khớp với bước 1, 2.

Trước khi chấm cả tập, cho 2 người cùng chấm khoảng 50 mẫu và tính độ đồng thuận (Cohen's κ). κ thấp nghĩa là rubric còn mơ hồ, cần sửa rubric rồi mới chấm tiếp. Người kiểm tra có thể sửa nhẹ nội dung reasoning. Mẫu đã sửa đánh dấu `edited=true`.

### 4.3. Cân bằng lớp

Tập train chỉ có 77 mẫu `text-sarcasm`. Nếu cân bằng mọi lớp về bằng lớp này thì sau khi lọc chỉ còn khoảng 100 mẫu. Vì vậy:

- **Giữ toàn bộ mẫu đạt rubric** của mọi lớp. Mất cân bằng được xử lý bằng trọng số trong loss phụ hoặc bằng sampler.
- Mẫu không đạt rubric vẫn dùng cho loss phân loại chính, chỉ không có reasoning.

### 4.4. Định dạng lưu

`rationale_dataset.json`, đặt trong `data/processed/rationales/` trên server (`CODEBASE.md` mục 7.2), và upload thành Kaggle dataset để nhóm dùng chung:

```json
{
  "meta": {"teacher": "Qwen/Qwen2.5-VL-7B-Instruct", "k": 3, "prompt_version": "v1", "rubric_version": "v1"},
  "samples": {
    "<id>": {
      "label": "multi-sarcasm",
      "text_sarcastic": false,
      "image_sarcastic": true,
      "rationale": "...",
      "n_correct": 2,
      "group": 1,
      "rationalized": false,
      "review": "pass",
      "edited": false
    }
  }
}
```

</details>

<details>
<summary>5. Train student</summary>

### 5.1. Phần chung của M1, M2, M3

**Encoder và fusion.** Dùng lại kiến trúc hiện có: PhoBERT + CLIP + OCR / mô tả ảnh (`multistream_clip` hoặc `cross_attn`), khoảng 135M + 150M tham số. Các phương pháp dùng cùng encoder, cùng fusion, cùng hyperparameter, chỉ khác phần head. Lúc suy luận **không cần reasoning hay teacher**, nên so sánh trực tiếp được với các baseline.

**Loss căn chỉnh reasoning**, chỉ tính trên các mẫu có reasoning đạt rubric:

```
L_rat = 1 − cos(proj(h_fusion), e_rationale)
```

- `h_fusion`: vector đầu ra của fusion, trước các head.
- `e_rationale`: embedding của reasoning, mã hoá bằng PhoBERT đóng băng, tính trước và lưu cache.

**Hai loại nhãn phụ**, mang nghĩa khác nhau:

| Nhãn phụ | Nghĩa | Nguồn | Có cho mẫu |
|---|---|---|---|
| `involves_text`, `involves_image` | châm biếm *có liên quan tới* text/ảnh | suy từ nhãn gốc: `label ∈ {multi, text}` và `label ∈ {multi, image}` | mọi mẫu train |
| `text_sarcastic`, `image_sarcastic` | text/ảnh *tự nó đã* mỉa mai khi đứng riêng | teacher (bước 1, 2 của prompt) | mẫu có reasoning đạt rubric |

Hai loại khác nhau ở mẫu `multi`. Với nhãn suy từ nhãn gốc, mọi mẫu `multi` đều là (có, có). Với nhãn của teacher, mẫu `multi` có thể là (có, có), (có, không), (không, có) hoặc (không, không), tương ứng 4 trường hợp ở mục 1. Thông tin này chỉ có teacher cung cấp được.

**Không suy nhãn 4 lớp từ hai nhãn phụ bằng luật cố định** như Approach 2 của Vu et al. Ví dụ, trường hợp 2 (text tự nó mỉa mai, ảnh thì không) có thể là `text-sarcasm` hoặc `multi`, tuỳ ảnh có góp phần hay không. Trường hợp 4 có cả hai nhãn phụ là "không" nhưng vẫn là `multi`. Nhãn phụ chỉ là tín hiệu giám sát hoặc đặc trưng trung gian, còn quyết định cuối do head học.

**Phương án thay thế** (chưa ưu tiên): một model seq2seq tiếng Việt cỡ base vừa sinh reasoning vừa đoán nhãn. Cách này chỉ nhìn được ảnh qua OCR và mô tả, và tốn công hơn nhiều.

### 5.2. M1: CoT distillation (đầu ra phẳng)

```
h_fusion ──► head 4 lớp (softmax)
         ├─► head T: text_sarcastic  (sigmoid, loss phụ)
         └─► head I: image_sarcastic (sigmoid, loss phụ)
```

```
L_M1 = CE_4lớp(tất cả mẫu)
     + λ_axis · [BCE(T, text_sarcastic) + BCE(I, image_sarcastic)]   (mẫu có nhãn của teacher)
     + λ_rat  · L_rat
```

Head T/I chỉ dùng lúc train để định hướng `h_fusion`, và bị bỏ đi lúc suy luận.

**Hiệu chỉnh prior:** cộng `log(prior_val) − log(prior_train)` vào 4 logit (logit adjustment). Prior lấy từ public test, vì đây là val khi `data.split=official`.

### 5.3. M2: CoT distillation + hierarchical (đầu ra phân cấp mềm)

```
                    h_fusion
          ┌────────────┼─────────────────────┐
   [T] text đứng riêng  [I] ảnh đứng riêng    [S] có châm biếm không
      (sigmoid)            (sigmoid)              (sigmoid)
          └─────┬──────┘                          │
                ▼                                 │
   [C] loại châm biếm | S: text / image / multi   │
      (softmax, đầu vào = [h_fusion; p_T; p_I])   │
                │                                 │
                ▼                                 ▼
   P(not)   = 1 − P(S)
   P(text)  = P(S) · P(text  | S)
   P(image) = P(S) · P(image | S)
   P(multi) = P(S) · P(multi | S)
```

- **Tầng S** học từ nhãn gốc trên toàn bộ 10,8k mẫu (`S = label ≠ not`). Hai lớp ở tầng này khá cân bằng (44/56 trong train).
- **Tầng C** chỉ phân biệt 3 loại châm biếm. Đầu vào gồm `h_fusion` và xác suất `p_T`, `p_I` của hai head T/I. Gradient không đi ngược qua `p_T`, `p_I` (dùng detach) để head T/I chỉ học theo teacher.
- **Tầng T và I** học theo nhãn `text_sarcastic` / `image_sarcastic` của teacher. Khác M1, ở đây T và I là đặc trưng trung gian có mặt cả lúc suy luận.

```
L_M2 = NLL_4lớp(P ở trên, tất cả mẫu)
     + λ_S    · BCE(S, label ≠ not)                                  (tất cả mẫu)
     + λ_axis · [BCE(T, text_sarcastic) + BCE(I, image_sarcastic)]   (mẫu có nhãn của teacher)
     + λ_rat  · L_rat
```

`NLL_4lớp` là cross-entropy trên xác suất 4 lớp đã nhân xác suất ở trên. Gradient đi qua cả S lẫn C, nên đây vẫn là một model train end-to-end. `λ_S` thêm giám sát trực tiếp cho tầng S và có thể bằng 0.

**Hiệu chỉnh prior theo từng tầng.** Label shift giữa train và test chủ yếu nằm ở tầng S:

| | train | public test | private test |
|---|---:|---:|---:|
| P(có châm biếm) | 43,9% | 68,7% | 69,5% |
| P(multi \| có châm biếm) | 89,1% | 97,0% | 95,5% |

Do đó chỉ cần dịch logit của head S một lượng `logit(prior_val) − logit(prior_train)`. Head C dịch thêm theo prior có điều kiện. Cách này đơn giản và ổn định hơn chỉnh cả 4 logit như M1.

**Lớp hiếm.** Ở tầng C, `text-sarcasm` chiếm 1,6% số mẫu có châm biếm (77/4 743), còn ở bài toán 4 lớp là 0,7%. Trọng số lớp hoặc focal loss áp dụng ở tầng C.

### 5.4. M3: CoT distillation + subclass (tách `multi` lúc train, gộp lúc suy luận)

**Ý tưởng.** Trong không gian `h_fusion`, lớp `multi` không phải một cụm mà gồm nhiều cụm. Mẫu thuộc trường hợp 2 nằm gần `text-sarcasm`, trường hợp 3 nằm gần `image-sarcasm`, còn trường hợp 4 có thể nằm sát `not-sarcasm` vì từng phần đứng riêng đều bình thường. Một head 4 lớp phải bao hết các cụm này bằng một vùng quyết định, và dễ bỏ rơi cụm nhỏ và khó (trường hợp 4). Đây lại chính là cụm đòi hỏi suy luận chéo modality nhiều nhất. M3 cho mỗi cụm một nhãn riêng lúc train. Kỹ thuật này gần với hidden stratification (Sohoni et al., 2020) và subclass distillation (Müller et al., 2020).

**Nhãn train (7 lớp).** Dùng `text_sarcastic` (T) và `image_sarcastic` (I) của teacher:

| Nhãn train | Điều kiện | Trường hợp (mục 1) | Gộp về |
|---|---|---|---|
| `not`, `text`, `image` | giữ nguyên | | giữ nguyên |
| `multi_TI` | T = 1, I = 1 | 1 | `multi` |
| `multi_T` | T = 1, I = 0 | 2 | `multi` |
| `multi_I` | T = 0, I = 1 | 3 | `multi` |
| `multi_none` | T = 0, I = 0 | 4 | `multi` |

```
h_fusion ──► head 7 lớp (softmax)
             │
             ▼  lúc suy luận
   P(multi) = P(multi_TI) + P(multi_T) + P(multi_I) + P(multi_none)
   argmax trên 4 lớp
```

**Gộp bằng cộng xác suất, không gộp bằng argmax.** Nếu lấy argmax trên 7 lớp rồi mới map về 4 lớp thì sẽ sai trong trường hợp như: 4 subclass mỗi cái 0,15–0,2, `text` 0,3. Khi đó argmax ra `text`, trong khi tổng P(multi) = 0,7. Nhầm lẫn giữa các subclass của `multi` không bị tính lỗi lúc suy luận. Các subclass chỉ có tác dụng định hình biểu diễn.

**Loss.** Chỉ mẫu `multi` có nhãn của teacher mới biết subclass. Mẫu `multi` còn lại không được đoán subclass, mà dùng loss ở mức thô (partial-label learning):

```
L_M3 = NLL_4lớp(P đã gộp, tất cả mẫu)
     + λ_sub · CE_7lớp(mẫu multi có nhãn của teacher)
     + λ_rat · L_rat
```

`NLL_4lớp` trên xác suất đã gộp tương đương `−log Σ P(multi_*)` với mẫu `multi`, nên mẫu chưa rõ subclass chỉ cần rơi vào nhóm `multi`, còn phân bố bên trong nhóm do model tự quyết. Có thể cài theo cùng mẹo của `HierarchicalHead`: head trả về log xác suất 4 lớp (gộp 4 logit `multi_*` bằng `logsumexp`), nên CE, `predict()` và class weight dùng lại được nguyên vẹn.

**So với M1, M2.**

| | Cách dùng nhãn T/I của teacher |
|---|---|
| M1 | head phụ tách rời, chỉ làm loss phụ, bỏ đi lúc suy luận |
| M2 | đặc trưng trung gian `p_T`, `p_I` làm đầu vào cho tầng C |
| M3 | ghép vào không gian nhãn: head học phân bố đồng thời của (nhãn, T, I) |

M3 là cách trực tiếp nhất. Model không bỏ qua được T/I, vì T/I là một phần của đáp án lúc train. M3 kết hợp được với M2: tầng C dùng 6 lớp (`text`, `image`, 4 subclass `multi`) thay cho 3.

**Các điểm cần xử lý.**
- **Subclass nhỏ.** Train có khoảng 4,2k mẫu `multi`. Sau khi lọc rubric rồi chia 4, `multi_I` và `multi_none` có thể chỉ còn vài trăm mẫu hoặc ít hơn. Cần xem phân bố 4 trường hợp (mục 6.5) trước khi train, và dùng trọng số lớp hoặc focal loss ở mức subclass.
- **Ranh giới `multi_T` với `text` vẫn khó.** Nhầm giữa hai lớp này vẫn bị tính lỗi sau khi gộp. M3 chỉ thêm tín hiệu trực tiếp cho ranh giới này, không xoá được nó.
- **Nhiễu của teacher.** Nhãn T/I sai thì subclass sai theo. Vì có bước gộp, subclass sai không làm sai nhãn cuối, chỉ kéo lệch biểu diễn.
- **Hiệu chỉnh prior.** Không biết prior subclass trên test. Chỉ hiệu chỉnh ở mức 4 lớp như M1, cộng cùng một lượng `log p_val(multi) − log p_train(multi)` vào cả 4 logit `multi_*`. Cách này giữ nguyên tỉ lệ giữa các subclass và dịch đúng tổng P(multi).
- Ở cột (b) của ma trận (nhãn phụ suy từ nhãn gốc), mọi mẫu `multi` đều là (1, 1), nên subclass suy biến thành 1 lớp. M3 chỉ có nghĩa từ cột (c) trở đi.

### 5.5. E1: kNN retrieval lúc suy luận

**Ý tưởng.** Sau khi fine-tune, dùng tập train làm bộ nhớ. Lúc suy luận, tìm các mẫu train giống nhất và kết hợp nhãn của chúng với dự đoán của model. Nền tảng tương tự kNN-LM (Khandelwal et al., 2020). Đây là bước hậu xử lý, không train lại, áp dụng được cho mọi model vì `forward_outputs` đã trả về vector `z` (chính là `h_fusion`).

```
1. Tính z cho toàn bộ train bằng checkpoint đã chọn (tắt dropout), chuẩn hoá L2 → bộ nhớ
2. Với mỗi mẫu cần đoán: lấy k láng giềng gần nhất theo cosine
3. w_c = Σ exp(sim_j / τ), lấy tổng trên các láng giềng thuộc lớp c
4. p_kNN(c) ∝ w_c · prior_val(c) / prior_train(c)
5. p_cuối = λ · p_model + (1 − λ) · p_kNN
```

- Bước 4 bắt buộc vì label shift: train chỉ có 43,9% mẫu châm biếm, còn public test là 68,7%, nên láng giềng thô thiên về `not`. So sánh với model phải dùng `p_model` đã hiệu chỉnh prior để hai bên cùng một phân bố đích. Đếm lớp được kẹp tối thiểu 1 như `compute_prior_shift`.
- Chọn `k`, `τ`, `λ` trên val theo cùng tiêu chí chọn checkpoint. Grid gồm `λ = 1` (không dùng retrieval) để luôn có mốc so sánh. Bộ nhớ chỉ chứa train, không thêm val vào khi chạm private test.
- Có thể in các láng giềng của một dự đoán để phân tích định tính.

**Hạn chế.**
- Hiệu quả phụ thuộc vào việc test có nhiều mẫu giống train không. Giả định "meme lặp template" chưa được đo. Nên báo cáo phân bố `sim` của láng giềng gần nhất trên val.
- `text-sarcasm` chỉ có 77 mẫu nên hiếm khi lọt vào top-k.
- Nếu `z` đang mã hoá shortcut thì láng giềng cũng theo shortcut đó.
- Checkpoint đã học thuộc train nên vector của mẫu train có thể tách lớp rõ hơn thực tế, làm `p_kNN` quá tự tin. `τ` và `λ` giúp giảm phần này, không loại bỏ hoàn toàn.
- Biến thể đưa láng giềng vào làm đầu vào lúc train không nằm trong phạm vi hiện tại, vì phức tạp hơn nhiều và phải tránh để model chép nhãn của chính mẫu đó.

### 5.6. B2: 3 model riêng + stacking

```
model T (chỉ caption)        ┐
model I (chỉ ảnh, OCR, mô tả)├─► dự đoán out-of-fold ─► classifier nhỏ ─► 4 lớp
model M (đa phương thức)     ┘
```

- **Không gộp bằng luật cố định,** vì (T = 1, I = 0) có thể là `text` hoặc `multi`, còn (T = 0, I = 0) vẫn có thể là `multi` (mục 5.1).
- **Bắt buộc dùng dự đoán out-of-fold** cho classifier gộp. Dự đoán trên chính dữ liệu đã train quá tự tin nên classifier sẽ học sai. Với 5 fold và 3 model là 15 lần train, cộng 3 lần train cuối trên toàn bộ train.
- **Nguồn nhãn:** bản đầu cho T, I, M cùng học nhãn 4 lớp nên không cần teacher. Bản học `text_sarcastic`, `image_sarcastic` chỉ có nhãn trên các mẫu đạt rubric.

Đây là baseline, không phải phương pháp chính. T và I không thấy tương tác giữa hai modality, nên trường hợp 4 dồn hết vào M, và cách này tốn k-fold mà không dùng chung biểu diễn như M1, M2. Giá trị của nó là đo xem end-to-end có hơn việc gộp muộn không. Riêng F1 của T và I đã có ích cho chẩn đoán ở mục 6.6.

### 5.7. E3: pretrain trên dữ liệu sarcasm ngoài

**Ý tưởng.** Dạy model khái niệm châm biếm trên một tập lớn trước, rồi fine-tune trên ViMMSD. Hướng này gần với intermediate-task transfer (Phang et al., 2018; Pruksachatkun et al., 2020) và domain-adaptive pretraining (Gururangan et al., 2020).

| Nguồn | Ưu | Nhược |
|---|---|---|
| MMSD2.0 (Qin et al., 2023), ảnh + text tiếng Anh, nhãn nhị phân | lớn, có ảnh, đã giảm shortcut hashtag/emoji của MMSD gốc | PhoBERT không đọc tiếng Anh: phải dịch máy hoặc đổi encoder đa ngôn ngữ, và lệch domain |
| Dữ liệu sarcasm tiếng Việt có nhãn | đúng ngôn ngữ | chưa xác định có tập công khai nào đủ lớn |
| Silver data: teacher gán nhãn bài đăng tiếng Việt chưa có nhãn | lớn tuỳ ý, dùng lại pipeline teacher | nhãn nhiễu, tốn GPU, phải tự thu thập |
| TAPT: train tiếp PhoBERT bằng MLM trên text mạng xã hội tiếng Việt | không cần nhãn | chỉ học văn phong, không học trực tiếp sarcasm |

**Cách ghép với M2.** Dữ liệu ngoài thường chỉ có nhãn có/không, nên pretrain backbone và tầng S trên nhãn đó, rồi fine-tune cả mô hình trên 4 lớp. Fine-tune phải dùng cùng cấu hình để so được với hàng phân cấp mềm.

**Hạn chế.** Định nghĩa sarcasm của tập ngoài có thể khác UIT DSC 2024 và gây chuyển giao âm. Chỉ dùng tập có bản quyền cho phép. Kết quả phải được so với cùng cấu hình không pretrain. Không được dùng text của public/private test để pretrain.

</details>

<details>
<summary>6. Đánh giá</summary>

### 6.1. Thiết lập

`data.split=official`. Báo cáo macro-F1, weighted-F1, F1 từng lớp và confusion matrix trên private test. Mốc tham chiếu là F1 0,41 của Vu et al. (2025), và khi so phải ghi rõ đang dùng loại F1 nào. Lớp `text-sarcasm` chỉ có 1 mẫu trong val và 7 mẫu trong test, nên không kết luận gì dựa trên F1 riêng của lớp này.

### 6.2. Ma trận thí nghiệm

Các phương pháp được so sánh theo cùng một thang. Mỗi cột thêm một thành phần so với cột bên trái:

| Cấu trúc đầu ra | (a) CE | (b) + nhãn phụ từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) + căn chỉnh reasoning (**đầy đủ**) |
|---|---|---|---|---|
| Phẳng 4 lớp | baseline | head T/I học `involves_*` | head T/I học `*_sarcastic` | **M1** |
| Phân cấp mềm | S + C | S + C, T/I học `involves_*` | S + C, T/I học `*_sarcastic` | **M2** |
| Subclass (7 lớp, gộp lúc suy luận) | – | – | 7 lớp theo T/I của teacher | **M3** |
| Phân cấp nối tiếp (tái lập Vu et al.) | Model 1 + Model 2 | – | – | – |

- Cột (a) và (b) không cần teacher, chạy được ngay.
- Mọi ô chạy có và không có hiệu chỉnh prior, với ít nhất 3 seed (dùng `data.split=official` nên split cố định, seed chỉ ảnh hưởng tới quá trình train).
- Thêm một ablation cho M1, M2, M3 ở cột (d): bỏ reasoning của nhóm 3, để đo tác dụng của bước rationalization.
- Hàng subclass chỉ có cột (c) và (d), vì ở (a) và (b) mọi mẫu `multi` đều cùng một subclass (mục 5.4).
- Các hướng ngoài ma trận: E1 áp dụng lên checkpoint tốt nhất của từng hàng, B2 và E3 so với ô tương ứng cùng backbone, D1 không tạo ô kết quả mới.

### 6.3. Câu hỏi cần trả lời

| So sánh | Câu hỏi |
|---|---|
| Phẳng (a) với phân cấp mềm (a) | Chỉ tách tầng S/C (chưa có teacher) đã giúp chưa? Phần lớn lợi ích dự kiến đến từ việc hiệu chỉnh label shift ở tầng S. |
| Phân cấp mềm (a) với nối tiếp (a) | Phân cấp mềm có tránh được lỗi lan truyền của Vu et al. không? |
| Cột (b) với cột (c) | Nhãn "text/ảnh đứng riêng" của teacher có thêm thông tin so với nhãn suy từ nhãn gốc không? |
| Cột (c) với cột (d) | Căn chỉnh với embedding reasoning có thêm gì ngoài nhãn phụ không? |
| **M1 với M2** | Đầu ra phân cấp có giúp student tận dụng reasoning tốt hơn không? Nên xem F1 từng lớp, nhất là `image-sarcasm`, và confusion giữa `multi` với `text`/`image`. |
| **M3 với M1, M2** | Đưa T/I của teacher vào không gian nhãn có tốt hơn dùng làm loss phụ (M1) hay đặc trưng trung gian (M2) không? Xem recall của `multi` trên val theo từng trường hợp 1–4, nhất là trường hợp 4. Trường hợp lấy từ T/I trong lần chạy zero-shot trên val của teacher. Các nhãn này chưa qua rubric nên chỉ để phân tích, và cần chấm một phần mẫu để ước lượng độ tin cậy. |
| **E1 với cùng model không retrieval** | Retrieval có thêm thông tin không? Xem cả `λ` được chọn và phân bố `sim`. |
| **M1, M2, M3 với B2** | Train end-to-end có hơn gộp muộn nhiều model không? |
| **E3 với cùng cấu hình không pretrain** | Dữ liệu ngoài giúp hay gây chuyển giao âm? |
| **E2 với zero-shot** | Few-shot có làm teacher tốt hơn và giảm nhóm 3 không? Reasoning của nó có đạt rubric nhiều hơn không? |
| **D1 trước và sau distillation** | Reasoning có làm model dùng ảnh nhiều hơn và bớt phụ thuộc tín hiệu bề mặt không? |

Nếu phân cấp mềm (a) đã hơn phẳng (a) gần bằng mức M2 hơn M1, thì lợi ích của M2 chủ yếu đến từ việc tách tầng để xử lý label shift, chưa phải từ reasoning.

### 6.4. Nhánh so sánh: reasoning làm đầu vào

Chạy teacher (không cho nhãn) trên cả public và private test, rồi đưa reasoning vào làm segment thứ 3 của PhoBERT, giống cách MiDRE làm. Phần chênh lệch giữa nhánh này và student ở mục 5 là cái giá của việc nén reasoning vào model nhỏ.

### 6.5. Kết quả phụ cần báo cáo

- Accuracy và F1 từng lớp của chính teacher trên tập train (lấy ở bước 3.3, trước khi lọc).
- Tỉ lệ mẫu mỗi nhóm, tỉ lệ đạt rubric theo lớp và theo nhóm, κ giữa hai người chấm.
- Phân bố 4 trường hợp của `multi` theo nhãn phụ của teacher, cũng là số mẫu mỗi subclass của M3.

### 6.6. D1: kiểm tra shortcut

Các phép đo này cho biết model phụ thuộc vào đâu. Chạy trên val, cho baseline tốt nhất trước, rồi lặp lại cho M1, M2, M3 để xem reasoning có làm giảm phụ thuộc vào shortcut không. Không dùng kết quả này để chọn model.

| Phép đo | Cách làm | Cách đọc |
|---|---|---|
| Text-only với đa phương thức | So F1 giữa `modality: text`, `image` và model đầy đủ | Đa phương thức không hơn text-only thì ảnh chưa đóng góp đáng kể |
| Đổi ảnh | Giữ caption, thay ảnh bằng ảnh của mẫu khác ngẫu nhiên, thay **cùng lúc** vector ảnh, OCR và mô tả | Tỉ lệ dự đoán đổi rất thấp thì model gần như bỏ qua ảnh |
| Bỏ tín hiệu bề mặt | Xoá emoji và dấu câu lặp trong caption | F1 giảm mạnh thì model phụ thuộc nhiều vào các tín hiệu này |
| Integrated Gradients | Xếp hạng token có đóng góp lớn nhất theo lớp | Top token chủ yếu là emoji hay từ cảm thán thì đó là dấu hiệu shortcut lexical |
| Attention map | Xem `last_attn` của `cross_attn` trên vài mẫu | Chỉ để minh hoạ định tính, attention không phải giải thích đáng tin cậy |

Lặp lại phép đổi ảnh với vài seed. Đổi ảnh làm dự đoán thay đổi chỉ cho thấy model **dùng** ảnh, không chứng minh nó học được sự lệch nghĩa giữa ảnh và text.

</details>

<details>
<summary>7. Rủi ro</summary>

| Rủi ro | Giảm thiểu |
|---|---|
| Teacher kém hơn baseline, reasoning không thêm thông tin | Thử trên 280 mẫu ở bước 3.1 rồi mới chạy cả tập |
| Rationalization bịa lý do cho khớp nhãn | Kiểm tra toàn bộ nhóm 3, so ablation có và không có nhóm 3 |
| Mô tả ảnh bị hallucination | Tiêu chí 2 của rubric. Đưa OCR vào prompt để teacher không phải tự đọc chữ |
| Rubric mơ hồ, người chấm không đồng ý với nhau | Đo κ trên 50 mẫu trước khi chấm cả tập |
| Chi phí GPU hoặc API | k = 3, cache để chạy tiếp, chạy thử trước khi chạy cả tập |
| Rò rỉ nhãn val/test | Chỉ lọc trên train. Nhánh 6.4 không dùng nhãn |
| M2: tầng S gánh hết label shift, chọn sai prior thì cả model lệch | Lấy prior từ val, báo cáo thêm kết quả không hiệu chỉnh. Kiểm tra độ nhạy theo prior (±10%) |
| M2: head C có ít dữ liệu cho `text`/`image` | Trọng số lớp hoặc focal ở tầng C. Báo cáo F1 có điều kiện trên các mẫu có châm biếm |
| M3: subclass `multi_I`, `multi_none` quá ít mẫu | Xem phân bố subclass trước khi train. Trọng số lớp ở mức subclass. Nếu một subclass quá nhỏ, gộp nó với subclass gần nhất |
| E1: chọn `k`, `τ`, `λ` quá khớp với val | Grid nhỏ, có `λ = 1`, chỉ chấm private test một lần ở bước cuối |
| E2: teacher chép nhãn ví dụ, `n_correct` mất ý nghĩa | Tách `prompt_version`, kiểm tra theo rubric, loại mẫu pilot khỏi nguồn ví dụ |
| E3: định nghĩa sarcasm khác nhau, chuyển giao âm, vướng bản quyền | Kiểm tra định nghĩa và giấy phép trước, luôn so với bản không pretrain |
| B2: rò rỉ qua dự đoán trên chính dữ liệu train | Chỉ fit classifier gộp trên dự đoán out-of-fold |

</details>

<details>
<summary>8. Code</summary>

| File | Nội dung |
|---|---|
| `src/vimmsd/models/heads.py` | `HierarchicalHead` (S × C, trả log xác suất 4 lớp), `CascadeHead`, `compute_prior_shift` theo tầng |
| `src/vimmsd/models/classifier.py` | `model.head` (`type`, `axis_heads`, `axis_targets`, `rationale_dim`), `forward_outputs`, `set_prior_shift` |
| `src/vimmsd/training/losses.py` | `SarcasmLoss`: CE/focal (= NLL với hierarchical), λ_S, λ_axis (gold / teacher), λ_rat, loss cascade |
| `src/vimmsd/training/trainer.py` | loss nhiều thành phần (log từng phần), `train.select_metric` / `select_exclude_labels` |
| `src/vimmsd/data/rationale.py` | prompt CoT 4 bước, `TeacherVLM`, parse JSON, phân nhóm, chọn mẫu chấm, κ, gộp `rationale_dataset.json` |
| `src/vimmsd/data/dataset.py` | `attach_rationales` (nhãn phụ teacher + embedding cho mẫu train đạt rubric), collator, `split_label_counts` |
| `scripts/make_id_lists.py`, `generate_rationales.py`, `eval_teacher.py` | giai đoạn 1–2 |
| `scripts/build_rationale_dataset.py` | `group` / `rationalize` / `export` / `kappa` / `merge` (giai đoạn 3–5) |
| `scripts/embed_rationales.py` | mã hoá reasoning bằng PhoBERT đóng băng, trung bình token |
| `scripts/evaluate.py` | `eval.prior_correction`: ghi thêm `<split>_metrics_prior.json` |
| `scripts/train.py` | `train.eval_test: false` để không chấm private test sau mỗi run |
| `scripts/run_experiments.py`, `summarize_results.py` | ma trận config × seed; `--extra-eval test` cho bước chấm cuối; cột `val+prior` |
| `configs/distill_*.yaml`, `configs/hier_cascade.yaml`, `configs/experiments/cot_distill*.yaml` | ma trận mục 6.2 |
| `tests/test_cot_distill.py` | head, loss, prior, Trainer thật với head mới, parse / phân nhóm / κ / gộp, dataset, config, plan |

Khác thiết kế ban đầu:
- **Cascade** (`hier_cascade.yaml`) chạy trong một model: head M và head R dùng chung backbone, head R chỉ học trên mẫu không phải multi, suy luận định tuyến cứng. Bài gốc dùng 2 model riêng và back-translation.
- **Embedding reasoning** lấy trung bình token của PhoBERT (không phải `<s>`), văn bản không kèm nhãn kết luận.
- **Nhãn phụ teacher thiếu**: mẫu not / text / image dùng nhãn suy từ nhãn gốc (hai nghĩa trùng nhau), mẫu multi thì bỏ.
- **M3** (mục 5.4) chưa có code. Cần thêm: `SubclassHead` trong `heads.py` (7 logit → log P 4 lớp bằng `logsumexp`, kèm `sub_logits`), `λ_sub` trong `SarcasmLoss` (mask như `axis_teacher_mask`), nhánh `subclass` trong `compute_prior_shift`, config `distill_sub_{c,d}.yaml` và thêm vào `cot_distill_grid.yaml`.
- **E1** chưa có code: dự kiến `scripts/knn_eval.py`.
- **E2** chưa có code: thêm tuỳ chọn few-shot và bước tìm ví dụ vào `data/rationale.py`, `scripts/generate_rationales.py`.
- **B2** chưa có code: cần chia k-fold trên train và `scripts/stack_predictions.py`.
- **E3** chưa có code: cần loader và config cho dữ liệu ngoài, nạp checkpoint pretrain trước khi fine-tune.
- **D1** chưa có code: dự kiến `scripts/diagnose_shortcuts.py`.
- **Nhánh 6.4** (reasoning làm đầu vào) chưa có code: `generate_rationales.py` sinh được reasoning cho public/private test, nhưng chưa có config đưa reasoning thành segment thứ 3.

</details>

## 3. Kế hoạch chạy

Chép nguyên từ `docs/experiment_plan.md`.

<details>
<summary>Tổng quan</summary>

| Giai đoạn | Việc | GPU | Cần teacher | Cổng |
|---|---|---|---|---|
| 0 | Chạy cột (a)/(b), kiểm tra shortcut (D1), kNN (E1) | có | không | G0 |
| 1 | Thử teacher, chốt prompt, thử few-shot (E2) | có | có | G1 |
| 2 | Sinh reasoning cho tập train | nhiều | có | – |
| 3 | Phân nhóm, rationalization, xuất file chấm | ít | có | – |
| 4 | Người kiểm tra theo rubric | không | không | G2 |
| 5 | Đóng gói `rationale_dataset.json` + embedding | ít | không | – |
| 6 | Code M3, train ma trận (cột c/d, M1, M2, M3), lặp lại D1 và E1 | nhiều | không | G3 |
| 7 | (tuỳ chọn) B3 reasoning làm đầu vào, B2 stacking, E3 pretrain | nhiều | tuỳ nhánh | G5 (E3) |
| 8 | Phân tích, viết báo cáo | không | không | G4 |

D1 và E1 không cần train lại, nhưng cần checkpoint của giai đoạn 0. Giai đoạn 0 và 1 **chạy song song được**: giai đoạn 0 không cần teacher, còn giai đoạn 1 không cần code student. Giai đoạn 4 (người chấm) thường là nút thắt về thời gian. Nên bắt đầu chấm ngay khi giai đoạn 3 xuất xong lô đầu (lớp hiếm trước), không đợi sinh xong cả tập.

</details>

<details>
<summary>Giai đoạn 0: nền móng, không cần teacher</summary>

**Mục tiêu.** Có head phân cấp chạy được và có số liệu cột (a)/(b) để làm mốc. Nhiều khả năng một phần lớn lợi ích của M2 đến từ đây (tách tầng S để xử lý label shift), nên cần đo riêng.

**Code** (local, có test):
1. `models/classifier.py`: thêm `head.type: flat | hierarchical`, gồm head S, C, T/I và phép nhân xác suất ra 4 lớp.
2. `training/losses.py`: NLL trên xác suất phân cấp, BCE cho head S và T/I.
3. `data/dataset.py`: thêm `involves_text`, `involves_image` suy từ nhãn gốc vào batch.
4. `training/metrics.py` và `scripts/evaluate.py`: hiệu chỉnh prior theo val, cho head phẳng (4 logit) và head phân cấp (theo tầng). Báo cáo cả có và không có hiệu chỉnh.
5. Configs: `distill_flat_a.yaml`, `distill_flat_b.yaml`, `distill_hier_a.yaml`, `distill_hier_b.yaml`, `hier_cascade.yaml`.
6. Test: kiểm tra xác suất phân cấp cộng lại bằng 1, gradient đi qua cả S và C, nhãn phụ suy đúng từ 4 nhãn.

**Chạy** (server): `python scripts/run_experiments.py --plan configs/experiments/cot_distill_stage0.yaml`, gồm 5 config × 3 seed, chia cho các GPU bằng `run_on_gpus`. Sau đó chạy `python scripts/summarize_results.py --plan ...` để có bảng trung bình ± độ lệch chuẩn.

**Trước giai đoạn 0**: chuẩn bị dữ liệu trên server (`CODEBASE.md` mục 7.4). Nên tải cache có sẵn `funalee/vimmsd-data-preprocessed` để kết quả so được với các run trên Kaggle của nhóm. Chỉ chạy lại `preprocess_text.py` / `extract_image_text.py` khi cố ý đổi cấu hình.

**Chẩn đoán và hậu xử lý** (sau khi có checkpoint, chỉ dùng val):
1. **D1, `scripts/diagnose_shortcuts.py`** (cần viết, có test): text-only với đa phương thức, đổi ảnh gồm cả OCR và mô tả, xoá tín hiệu bề mặt, Integrated Gradients (`experiment_design.md` mục 6.6). Text-only lấy từ plan `phase4_baselines`. Ghi vào `outputs/diagnostics/`.
2. **E1, `scripts/knn_eval.py`** (cần viết, có test): nạp checkpoint tốt nhất của mỗi hàng, tính `z` của train và val, dò `k` ∈ {5, 10, 20, 50}, `τ` ∈ {0,05; 0,1; 0,2}, `λ` ∈ {0,5; 0,7; 0,9; 1} theo tiêu chí chọn checkpoint (mục 5.5). Ghi vào `outputs/knn/`, kèm phân bố `sim` của láng giềng gần nhất.

**Đầu ra.** Bảng macro-F1, weighted-F1 và F1 từng lớp trên **val**, có và không có hiệu chỉnh prior. Báo cáo D1. Bảng E1 so với cùng checkpoint không retrieval.

**Cổng G0.**
- Head phân cấp không kém head phẳng quá 1 điểm macro-F1 trên val. Nếu kém hơn, kiểm tra lại code trước khi đi tiếp.
- Ghi lại số tốt nhất của cột (a)/(b). Đây là mốc mà teacher ở G1 phải vượt. Mốc này không tính E1, vì teacher cũng không dùng retrieval.
- D1 và E1 không chặn cổng này. Nếu đổi ảnh gần như không làm dự đoán đổi, ưu tiên xem lại fusion trước khi đầu tư nhiều vào teacher. E1 chỉ được giữ nếu chênh lệch trên val lớn hơn 1 độ lệch chuẩn giữa các seed và `λ` được chọn nhỏ hơn 1.

</details>

<details>
<summary>Giai đoạn 1: thử teacher</summary>

**Mục tiêu.** Chọn teacher và chốt prompt trước khi tốn GPU cho cả tập train.

**Tập thử.** Lấy từ **train**: 70 mẫu mỗi lớp, tổng 280 mẫu (`text-sarcasm` có 77 mẫu nên gần như lấy hết). Teacher chỉ được đánh giá chứ không học trên các mẫu này, nên lấy từ train không gây rò rỉ. Không lấy từ val được vì val chỉ có 1 mẫu `text-sarcasm` và 28 mẫu `image-sarcasm`. Chia tập thử thành:
- **prompt-dev**: 40 mẫu mỗi lớp, dùng để sửa prompt.
- **pilot-eval**: 30 mẫu mỗi lớp, chỉ chạy một lần với prompt đã chốt để báo cáo.

**Code.**
1. `scripts/generate_rationales.py`: dùng lại `VLMDescriber` (backend `hf`) và `build_image_text_cache` của `data/image_text.py` để có sẵn khả năng chạy tiếp, chia shard và gộp shard. Tham số: `--k`, `--temperature`, `--prompt-version`, `--ids-file`, `--with-label` (dùng cho rationalization). Mỗi lần sinh lưu một dòng JSON, parse lỗi thì ghi `parse_error` chứ không bỏ.
2. `scripts/eval_teacher.py`: so output của teacher với nhãn, ghi confusion matrix, F1 từng lớp, tỉ lệ parse lỗi và thời gian mỗi mẫu vào `outputs/cot_distill/stage1/`. Script này thay cho notebook.
3. `scripts/make_id_lists.py`: tạo `pilot_dev.txt`, `pilot_eval.txt`, `batch{1..4}.txt` theo seed cố định, lưu ở `data/processed/rationales/ids/`.

**Chạy.**
1. Đối chiếu định nghĩa 4 nhãn với hướng dẫn gán nhãn của UIT DSC 2024, rồi viết prompt v1 theo khung 4 bước (`experiment_design.md` mục 3.2).
2. Chạy Qwen2.5-VL-7B-Instruct với k = 1 trên prompt-dev, sửa prompt 2–3 vòng.
3. Chạy prompt đã chốt với k = 3 trên pilot-eval.
4. Chạy zero-shot (không cho nhãn, k = 1) trên **toàn bộ val** (1 413 mẫu) để so trực tiếp với mốc G0. Ở đây chỉ đánh giá, không chọn mẫu, nên không rò rỉ. Giữ lại T/I của lần chạy này để phân tích recall của `multi` theo trường hợp (mục 6.3 của thiết kế).
5. Nếu Qwen 7B không qua G1: thử bản 32B (cần lượng tử hoá trên 2×T4), hoặc GPT-4o / Gemini trên đúng 280 mẫu đó, kèm ước tính chi phí cho cả tập.
6. **E2 (few-shot, cần viết code):** sau khi prompt zero-shot ổn định, so với bản few-shot trên prompt-dev (`experiment_design.md` mục 3.4). Bản được chọn mới chạy bước 3 và 4, và ghi vào `prompt_version` (ví dụ `v1_fewshot`).

**Đầu ra.** Prompt chốt (`prompt_version=v1`), tên teacher, tốc độ (mẫu/giờ trên 2×T4), tỉ lệ parse lỗi.

**Cổng G1** (đánh giá trên val, cho bản prompt đã chọn):
- F1 của teacher ở `image-sarcasm` **hoặc** `multi-sarcasm` cao hơn mốc tốt nhất của G0. Nếu teacher kém hơn baseline ở cả hai lớp này thì reasoning của nó khó thêm được thông tin. Khi đó dừng hướng distillation, chỉ báo cáo teacher như một baseline zero-shot (README mục 5.4).
- Tỉ lệ parse lỗi dưới 5%.
- Dùng tốc độ đo được để ước tính thời gian cho giai đoạn 2. Nếu quá lâu so với thời hạn, giảm phạm vi theo thứ tự ưu tiên ở giai đoạn 2, hoặc chuyển sang chạy teacher bằng `vllm`.

</details>

<details>
<summary>Giai đoạn 2: sinh reasoning cho tập train</summary>

**Phạm vi, theo thứ tự ưu tiên.** Chạy theo thứ tự này để nếu dừng giữa chừng vẫn có dữ liệu dùng được:

| Lô | Mẫu | Số mẫu | k |
|---|---|---:|---:|
| 1 | toàn bộ `text-sarcasm` + `image-sarcasm` | 519 | 5 |
| 2 | toàn bộ `multi-sarcasm` | 4 224 | 3 |
| 3 | `not-sarcasm` lấy ngẫu nhiên theo seed cố định | 2 000 | 3 |
| 4 | phần `not-sarcasm` còn lại | 4 062 | 3 (nếu còn quota) |

Hai lớp hiếm dùng k = 5 vì mỗi mẫu có giá trị cao, và lô 1 chỉ có 519 mẫu.

**Chạy.** Trên server, mỗi GPU một tiến trình `generate_rationales.py --num-shards N --shard i` trong `tmux`, sinh không cho nhãn. Bị dừng giữa chừng thì chạy lại đúng lệnh đó, mẫu đã có trong `raw/<prompt_version>/` sẽ được bỏ qua. Nếu teacher cần nhiều VRAM hơn một GPU (ví dụ bản 32B), dùng `device_map=auto` trên nhiều GPU và chạy 1 shard.

**Đầu ra.** `rationales_raw/*.jsonl`, mỗi dòng gồm `id`, `sample_idx`, `output`, `parse_error`, `prompt_version`, `teacher`.

</details>

<details>
<summary>Giai đoạn 3: phân nhóm, rationalization, xuất file chấm</summary>

**Code.** `scripts/build_rationale_dataset.py`, gồm 3 lệnh con:
- `group`: tính `n_correct / k`, gán nhóm 1/2/3 (`experiment_design.md` mục 4.1), chọn reasoning đại diện.
- `rationalize`: liệt kê id nhóm 3, gọi `generate_rationales.py --with-label`.
- `export`: xuất `review/<lô>.csv` với các cột `id`, `label`, `caption`, `ocr`, `image_path`, `rationale`, `group`, `rationalized`, và 4 cột rubric để trống.

**Chạy.** Chạy rationalization cho nhóm 3 trên server. Số lần gọi bằng số mẫu nhóm 3, với k = 1.

**Đầu ra.** File CSV để chấm, theo lô, và thống kê số mẫu từng nhóm theo lớp. Nên báo cáo thống kê này ở mục 6.5 của tài liệu thiết kế.

</details>

<details>
<summary>Giai đoạn 4: người kiểm tra</summary>

**Khối lượng.** Gồm toàn bộ nhóm 2, toàn bộ nhóm 3 và 15% nhóm 1. Ước tính lại sau giai đoạn 3 vì phụ thuộc độ chính xác của teacher.

**Các bước.**
1. **Hiệu chỉnh rubric**: 2 người chấm độc lập cùng 50 mẫu, gồm cả 4 lớp và cả 3 nhóm. Tính Cohen's κ cho từng tiêu chí.
2. Nếu có tiêu chí κ < 0,6: họp thống nhất, sửa mô tả rubric (`rubric_version` v2) rồi chấm lại 50 mẫu khác.
3. Chia các lô CSV cho 4 thành viên. Chấm lớp hiếm trước (lô 1). Mỗi người chấm cả mẫu nhóm 1 lẫn nhóm 3 để tránh lệch theo người.
4. Mẫu ghi `edited=true` nếu người chấm sửa reasoning. Không sửa nhãn gốc.

**Cổng G2.**
- κ ≥ 0,6 trên mọi tiêu chí sau khi hiệu chỉnh.
- Mỗi lớp có ít nhất khoảng 30 mẫu đạt rubric. Lớp `text-sarcasm` có thể không đạt mức này. Nếu thiếu, ghi lại thành hạn chế và vẫn đi tiếp, vì loss phụ có trọng số lớp và loss chính vẫn dùng đủ 10,8k mẫu.
- Tỉ lệ đạt rubric của nhóm 1 (lấy mẫu ngẫu nhiên) ≥ 80%. Nếu thấp hơn, chọn reasoning đúng nhiều lần cũng chưa đủ tin cậy, nên chấm thêm nhóm 1.

</details>

<details>
<summary>Giai đoạn 5: đóng gói dữ liệu</summary>

1. `build_rationale_dataset.py merge`: gộp kết quả chấm thành `rationale_dataset.json` theo định dạng ở `experiment_design.md` mục 4.4.
2. Mã hoá reasoning bằng PhoBERT đóng băng thành `rationale_embeddings.npy` cùng danh sách id. Tính trước một lần để không phải chạy PhoBERT trên reasoning mỗi epoch.
3. Lưu ở `data/processed/rationales/` trên server. Thêm `data.rationale.dir` / `file` vào `base.yaml`, theo cách đang làm với `image_text`. Đồng thời upload thành **Kaggle dataset** riêng, ví dụ `vimmsd-rationales`, phiên bản ứng với `prompt_version` / `rubric_version`, để nhóm dùng lại được trên Kaggle.
4. Code: `data/dataset.py` đọc file, thêm `text_sarcastic`, `image_sarcastic`, `e_rationale` và mask "có reasoning" vào batch. Có test cho mẫu không có reasoning.
5. Thống kê số mẫu mỗi subclass `multi_*`. Subclass dưới khoảng 30 mẫu đạt rubric thì gộp với subclass gần nhất (mục 5.4), ghi quyết định lại trước khi train M3.

</details>

<details>
<summary>Giai đoạn 6: train ma trận thí nghiệm</summary>

**Code M3** (local, có test, làm trước khi chạy ma trận): `SubclassHead`, `λ_sub`, nhánh `subclass` của `compute_prior_shift`, config (`experiment_design.md` mục 8). Test gồm: xác suất gộp đúng bằng tổng 4 subclass và cộng lại bằng 1, mẫu `multi` thiếu nhãn teacher chỉ chịu loss mức thô, hiệu chỉnh prior giữ nguyên tỉ lệ giữa các subclass.

**Configs.** Gồm `distill_flat_{c,d}.yaml`, `distill_hier_{c,d}.yaml`, `distill_sub_{c,d}.yaml` và `distill_{flat,hier,sub}_d_nogroup3.yaml`. Kết hợp với cột (a)/(b) và cascade ở giai đoạn 0 là đủ ma trận ở `experiment_design.md` mục 6.2.

**Chọn hyperparameter trên val.** Grid nhỏ cho `λ_axis` ∈ {0,1; 0,3; 1} và `λ_rat` ∈ {0,1; 0,3; 1}, chỉ chạy trên cột (d), 1 seed. M3 dò riêng `λ_sub` ∈ {0,1; 0,3; 1} thay cho `λ_axis`. Sau đó cố định giá trị tốt nhất cho mọi ô.

**Chạy chính thức.** Mỗi ô 3 seed, tiêu chí chọn checkpoint giống nhau cho mọi ô. Không chọn theo macro-F1 4 lớp trên val, vì `text-sarcasm` chỉ có 1 mẫu trong val. Nên dùng weighted-F1, hoặc macro-F1 trên 3 lớp còn lại. Chốt tiêu chí trước khi chạy và ghi vào `reports/results.md`.

**Lặp lại trên val.** Chạy D1 cho checkpoint tốt nhất của M1, M2, M3 để so với baseline. Chạy E1 và chốt `k`, `τ`, `λ` cho từng hàng trước bước đánh giá cuối.

**Đánh giá cuối.** Chỉ sau khi mọi ô đã chọn xong checkpoint mới chạy `scripts/evaluate.py --split test` (private test) cho toàn bộ ma trận một lần, kèm phiên bản có E1 với siêu tham số đã chốt.

**Cổng G3.**
- Mọi ô đã đủ 3 seed. Báo cáo trung bình ± độ lệch chuẩn.
- M1 hoặc M2 hơn ô (b) cùng hàng, hoặc M3 hơn ô (b) của hàng phẳng, **quá 1 độ lệch chuẩn** thì mới kết luận reasoning có tác dụng. Nếu không, báo cáo đó là kết quả âm tính, và vẫn giữ phân tích ở mục 6.3 của tài liệu thiết kế.

</details>

<details>
<summary>Giai đoạn 7 (tuỳ chọn): các nhánh mở rộng</summary>

Chỉ làm khi còn thời gian và quota GPU sau giai đoạn 6, theo thứ tự B3, B2, E3. Mỗi nhánh đều dùng checkpoint và siêu tham số chọn trên val, rồi mới chấm private test một lần.

### 7A. B3: reasoning làm đầu vào

Chạy teacher không cho nhãn trên public và private test (khoảng 2,9k mẫu, k = 1). Đưa reasoning vào làm segment thứ 3 của PhoBERT, train trên reasoning của train (lấy reasoning đại diện của mẫu, kể cả mẫu không đạt rubric, cho giống điều kiện lúc test). So với M1/M2 để đo cái giá của việc nén reasoning vào student.

### 7B. B2: 3 model riêng + stacking

1. Code: chia 5 fold trên train, lưu dự đoán out-of-fold, `scripts/stack_predictions.py`, có test bảo đảm không mẫu nào được dự đoán bởi model đã học nó.
2. Chạy model T, I, M trên 5 fold (15 run, 1 seed), rồi train lại mỗi model trên toàn bộ train.
3. Fit classifier gộp trên dự đoán out-of-fold, chọn siêu tham số trên val, rồi so với M1, M2, M3.

### 7C. E3: pretrain trên dữ liệu ngoài

1. Khảo sát nguồn dữ liệu: định nghĩa nhãn, giấy phép, số mẫu, mức trùng lặp với ViMMSD. Bản đầu nên dùng MMSD2.0 cho backbone và tầng S của M2, nếu dịch máy đạt yêu cầu khi đọc thử.
2. Code: loader cho dữ liệu ngoài, train nhị phân, nạp checkpoint trước khi fine-tune, có test.
3. Pretrain, rồi fine-tune cùng cấu hình với ô phân cấp mềm tương ứng, 3 seed.

**Cổng G5.** E3 hơn cùng cấu hình không pretrain quá 1 độ lệch chuẩn trên val thì mới đưa vào đánh giá cuối. Nếu không, báo cáo kết quả âm tính. Nếu không có nguồn dữ liệu đạt yêu cầu ở bước 1 thì dừng nhánh này và ghi lại lý do.

</details>

<details>
<summary>Giai đoạn 8: phân tích và báo cáo</summary>

**Cổng G4**: báo cáo đủ các mục sau.
1. Bảng ma trận đầy đủ trên private test (cả có và không có hiệu chỉnh prior), so với mốc 0,41 của Vu et al., ghi rõ loại F1.
2. Trả lời từng câu hỏi ở `experiment_design.md` mục 6.3.
3. Kết quả của teacher: zero-shot trên val, độ chính xác trên train theo lớp, phân bố 4 trường hợp của `multi`.
4. Thống kê lọc và chấm: tỉ lệ từng nhóm, tỉ lệ đạt rubric, κ.
5. 5–10 ví dụ định tính, gồm mẫu M2 hoặc M3 đúng mà baseline sai và ngược lại, kèm reasoning của teacher.
6. Kết quả D1 trước và sau distillation, E1 so với không retrieval, E2 so với zero-shot, và các nhánh tuỳ chọn đã chạy, kể cả nhánh dừng giữa chừng.
7. Hạn chế: số mẫu `text-sarcasm`, độ nhạy theo prior, chi phí.

Kết quả ghi vào `reports/results.md`. Bản tổng hợp riêng đặt ở `reports/experiment_report.md`.

</details>

<details>
<summary>Theo dõi tiến độ</summary>

| # | Việc | Phụ thuộc | Trạng thái |
|---|---|---|---|
| S.1 | Cài server (`setup_env.sh`), tải data + cache có sẵn (`prepare_data.sh`), chạy plan `smoke` | – | đã có script, chưa chạy trên server |
| S.2 | `run_experiments.py`, `summarize_results.py`, `preprocess_text.py` | – | xong (có test), chưa chạy trên server |
| 0.1 | Head phân cấp + loss + test | – | có code + test |
| 0.2 | Nhãn phụ từ nhãn gốc + hiệu chỉnh prior | – | có code + test |
| 0.3 | Chạy plan `phase4_baselines` (mốc), rồi (a)/(b) phẳng, phân cấp, cascade × 3 seed | S.1, S.2, 0.1, 0.2 | chưa làm |
| 0.4 | D1: `diagnose_shortcuts.py` + test, chạy trên baseline | 0.3 | có code + test, chưa chạy |
| 0.5 | E1: `knn_eval.py` + test, chạy trên baseline | 0.3 | có code + test |
| 1.1 | Đối chiếu định nghĩa nhãn, viết prompt v1 | – | chưa làm |
| 1.2 | `generate_rationales.py`, `eval_teacher.py`, `make_id_lists.py` | – | có code + test |
| 1.3 | prompt-dev, pilot-eval, zero-shot trên val → G1 | 1.1, 1.2, 0.3 | chưa làm |
| 1.4 | E2: tuỳ chọn few-shot + test, so với zero-shot trên prompt-dev | 1.2 | chưa có code |
| 2.1 | Sinh lô 1–3 | G1 | chưa làm |
| 3.1 | `build_rationale_dataset.py` group/rationalize/export | 2.1 (lô 1) | có code + test |
| 4.1 | Hiệu chỉnh rubric (50 mẫu, κ) | 3.1 | có script κ, chưa chấm |
| 4.2 | Chấm các lô | 4.1 | chưa làm |
| 5.1 | `rationale_dataset.json` + `embed_rationales.py` + upload Kaggle dataset | 4.2 | có script, chưa chạy |
| 5.2 | Dataset đọc reasoning + test | – | có code + test |
| 5.3 | Thống kê và quyết định gộp subclass | 5.1 | chưa làm |
| 6.0 | M3: head, loss, prior, config + test | 5.3 cho quyết định gộp | chưa có code |
| 6.1 | Grid λ trên val | 5.1, 5.2, 6.0 | chưa làm |
| 6.2 | Ma trận đầy đủ × 3 seed, D1 + E1 trên val, đánh giá private test → G3 | 6.1, 0.4, 0.5 | chưa làm |
| 7.1 | (tuỳ chọn) B3: reasoning làm đầu vào | 6.2 | chưa có code |
| 7.2 | (tuỳ chọn) B2: k-fold + stacking | 6.2 | chưa có code |
| 7.3 | (tuỳ chọn) E3: khảo sát dữ liệu → pretrain → G5 | 6.2 | chưa có code |
| 8.1 | Báo cáo → G4 | 6.2 | chưa làm |

</details>

## 4. Dữ liệu

| Mục | Giá trị |
|---|---|
| num_images | 13722 |
| ocr_model | PaddleOCR (phát hiện vùng chữ) + VietOCR vgg_seq2seq |
| vlm_model | 5CD-AI/Vintern-1B-v3_5 |
| missing | {'ocr': 0, 'description': 0} |

## 5. Giai đoạn 0: baseline và chẩn đoán

### 5.1. Baseline của nhóm trên Kaggle

| Cấu hình | Commit | Val macro-F1 | Test macro-F1 | Thời gian (phút) | Ghi chú |
|---|---|---|---|---|---|
| `image_only` | `646475c` | 0.3323 | 0.3422 | 8.5 | Task 5: CLIP Image-only \| peak VRAM 1.67 GB |
| `image_only_vit` | `646475c` | 0.3267 | 0.3313 | 10.5 | Task 6: ViT Image-only \| peak VRAM 2.33 GB |
| `vlm_image_centric` (Vintern, không metadata) | `646475c` | n/a | 0.0453 | ~7.0 | Task 7: zero-shot inference 1.504 mẫu \| invalid rate 0.0000 |
| `vlm_image_centric` (Vintern, có OCR + mô tả) | `646475c` | n/a | 0.0151 | ~7.3 | Task 7: zero-shot inference 1.504 mẫu \| invalid rate 0.0000 |

### 5.2. Cột (a)/(b) trên val, chưa hiệu chỉnh prior

| Cấu hình | Số seed | Macro-F1 | Macro-F1 bỏ text | Weighted-F1 | Accuracy | F1 `not-sarcasm` | F1 `text-sarcasm` | F1 `image-sarcasm` | F1 `multi-sarcasm` |
|---|---|---|---|---|---|---|---|---|---|
| `distill_flat_a` | 3 | 0.391 ± 0.011 | 0.521 ± 0.015 | 0.760 ± 0.022 | 0.758 ± 0.027 | 0.657 ± 0.030 | 0.000 ± 0.000 | 0.077 ± 0.069 | 0.830 ± 0.022 |
| `distill_flat_b` | 3 | 0.389 ± 0.014 | 0.518 ± 0.019 | 0.763 ± 0.010 | 0.771 ± 0.016 | 0.654 ± 0.017 | 0.000 ± 0.000 | 0.064 ± 0.061 | 0.836 ± 0.015 |
| `distill_hier_a` | 3 | 0.379 ± 0.003 | 0.506 ± 0.004 | 0.757 ± 0.016 | 0.766 ± 0.023 | 0.648 ± 0.006 | 0.000 ± 0.000 | 0.040 ± 0.035 | 0.830 ± 0.025 |
| `distill_hier_b` | 3 | 0.384 ± 0.006 | 0.512 ± 0.008 | 0.754 ± 0.011 | 0.758 ± 0.013 | 0.655 ± 0.025 | 0.000 ± 0.000 | 0.058 ± 0.053 | 0.821 ± 0.008 |
| `hier_cascade` | 3 | 0.380 ± 0.006 | 0.507 ± 0.008 | 0.764 ± 0.009 | 0.771 ± 0.016 | 0.671 ± 0.019 | 0.000 ± 0.000 | 0.018 ± 0.030 | 0.832 ± 0.019 |

### 5.3. Cột (a)/(b) trên val, có hiệu chỉnh prior

| Cấu hình | Số seed | Macro-F1 | Macro-F1 bỏ text | Weighted-F1 | Accuracy | F1 `not-sarcasm` | F1 `text-sarcasm` | F1 `image-sarcasm` | F1 `multi-sarcasm` |
|---|---|---|---|---|---|---|---|---|---|
| `distill_flat_a` | 3 | 0.373 ± 0.006 | 0.498 ± 0.008 | 0.755 ± 0.010 | 0.773 ± 0.011 | 0.603 ± 0.019 | 0.000 ± 0.000 | 0.042 ± 0.043 | 0.848 ± 0.007 |
| `distill_flat_b` | 3 | 0.366 ± 0.008 | 0.488 ± 0.011 | 0.760 ± 0.011 | 0.783 ± 0.008 | 0.612 ± 0.030 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.853 ± 0.004 |
| `distill_hier_a` | 3 | 0.354 ± 0.013 | 0.472 ± 0.017 | 0.742 ± 0.016 | 0.769 ± 0.008 | 0.572 ± 0.048 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.845 ± 0.002 |
| `distill_hier_b` | 3 | 0.353 ± 0.010 | 0.471 ± 0.014 | 0.741 ± 0.014 | 0.769 ± 0.010 | 0.565 ± 0.038 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.847 ± 0.005 |
| `hier_cascade` | 3 | 0.358 ± 0.004 | 0.477 ± 0.005 | 0.748 ± 0.003 | 0.774 ± 0.005 | 0.584 ± 0.023 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.848 ± 0.007 |

### 5.4. Mốc cho G1

| Lớp | Hiệu chỉnh prior | F1 tốt nhất | Cấu hình |
|---|---|---|---|
| `image-sarcasm` | không | 0.077 | `distill_flat_a` |
| `multi-sarcasm` | không | 0.836 | `distill_flat_b` |
| `image-sarcasm` | có | 0.042 | `distill_flat_a` |
| `multi-sarcasm` | có | 0.853 | `distill_flat_b` |

### 5.5. D1: kiểm tra shortcut trên baseline

| Run | Macro-F1 bỏ text | Sau đổi ảnh | Đổi nhãn khi đổi ảnh | Dịch xác suất | Sau khi bỏ emoji/dấu câu | Đổi nhãn trên caption bị sửa | Sai số IG |
|---|---|---|---|---|---|---|---|
| `distill_flat_a_s1` | 0.537 | 0.502 | 0.143 | 0.139 | 0.519 | 0.096 | 0.208 |
| `distill_flat_a_s2` | 0.520 | 0.441 | 0.280 | 0.272 | 0.482 | 0.188 | 0.189 |
| `distill_flat_a_s42` | 0.507 | 0.484 | 0.143 | 0.108 | 0.495 | 0.067 | 0.025 |

Theo lớp:

| Run | Lớp | F1 gốc | F1 sau đổi ảnh | Đổi nhãn khi đổi ảnh | F1 sau khi bỏ emoji/dấu câu |
|---|---|---|---|---|---|
| `distill_flat_a_s1` | `not-sarcasm` | 0.663 | 0.620 | 0.219 | 0.655 |
| `distill_flat_a_s1` | `image-sarcasm` | 0.102 | 0.055 | 0.214 | 0.071 |
| `distill_flat_a_s1` | `multi-sarcasm` | 0.846 | 0.830 | 0.105 | 0.832 |
| `distill_flat_a_s2` | `not-sarcasm` | 0.624 | 0.521 | 0.333 | 0.607 |
| `distill_flat_a_s2` | `image-sarcasm` | 0.130 | 0.027 | 0.452 | 0.061 |
| `distill_flat_a_s2` | `multi-sarcasm` | 0.805 | 0.777 | 0.249 | 0.777 |
| `distill_flat_a_s42` | `not-sarcasm` | 0.683 | 0.641 | 0.155 | 0.659 |
| `distill_flat_a_s42` | `image-sarcasm` | 0.000 | 0.000 | 0.167 | 0.000 |
| `distill_flat_a_s42` | `multi-sarcasm` | 0.838 | 0.811 | 0.136 | 0.826 |

Từ có điểm IG cao nhất trong caption, theo lớp dự đoán:

| Run | Lớp dự đoán | Tỉ lệ điểm ở dấu câu | Từ |
|---|---|---|---|
| `distill_flat_a_s1` | `not-sarcasm` | 27.4% | minh tú, ông, trái tim, lửa, siêu mẫu, tim, đăng, chúng tôi, cuối cùng, voz, xem, yêu |
| `distill_flat_a_s1` | `multi-sarcasm` | 25.7% | vậy, interpool, cười, ?, ), :, có, đi, con, là, đã, cho |
| `distill_flat_a_s1` | `image-sarcasm` | 20.5% | voz, -, :, về, ?, ", đã, ', ,, ông, người, (trắng) |
| `distill_flat_a_s2` | `not-sarcasm` | 20.5% | voz, nhé, f, (trắng), mỗi, đừng, cả, vẫn, ông, mùa, 6, trái tim |
| `distill_flat_a_s2` | `multi-sarcasm` | 21.5% | interpool, bạn, cũng, mặt, không, ), ,, ?, =, người, tôi, này |
| `distill_flat_a_s2` | `image-sarcasm` | 29.3% | =, voz, của, (trắng), 33, ,, f, ?, :, ', ), - |
| `distill_flat_a_s42` | `not-sarcasm` | 17.9% | đặc biệt, voz, (trắng), đang, nào, cao, diễn viên, sao, tổ chức, 2, ông, thế giới |
| `distill_flat_a_s42` | `multi-sarcasm` | 29.8% | interpool, :, cũng, ), cười, không, =, ?, cái, ,, được, nói |

Kiểm tra trên 3 seed của `distill_flat_a`, mỗi seed 3 cách ghép ngẫu nhiên, đổi cùng lúc ảnh, OCR và mô tả. So sánh text-only chưa có vì đang chờ kết quả Kaggle; attention map chỉ có với `cross_attn`.

- Model có dùng ảnh: trung bình 18,9% dự đoán đổi khi đổi ảnh, nên điều kiện "xem lại fusion" của kế hoạch không bị kích hoạt. Đổi nhãn không chứng minh model học được sự lệch nghĩa giữa ảnh và text.
- Caption gánh phần lớn: macro-F1 bỏ `text-sarcasm` chỉ giảm từ 0,521 xuống 0,476 khi ảnh sai. `multi-sarcasm` giảm 0,016–0,028, trong khi luôn đoán `multi-sarcasm` trên val đã đạt F1 khoảng 0,80.
- `image-sarcasm` phụ thuộc ảnh: seed 1 giảm từ 0,102 xuống khoảng 0,055, seed 2 từ 0,130 xuống khoảng 0,027; seed 42 không dự đoán lớp này. Val chỉ có 28 mẫu lớp này.
- Emoji và dấu câu lặp ảnh hưởng ít: giảm khoảng 2,2 điểm.
- Điểm IG ở mức định tính: sai số completeness với 300 bước là 0,025 ở seed 42 và khoảng 0,2 ở hai seed còn lại.
- Nghi vấn shortcut theo nguồn (chưa kết luận): `interpool` và `voz` đứng đầu ở cả 3 seed. Caption chứa `interpool`: train 468 bài, 59% `multi-sarcasm` (toàn tập 39%); val 90 bài, 80% (toàn tập 67%). Caption chứa `voz`: train 222 bài, 12% `image-sarcasm` (toàn tập 4%); val 105 bài, 60% `not-sarcasm` (toàn tập 31%).

### 5.6. Kiểm tra thẻ nguồn trong caption

| Run | Cách xoá | Caption bị đổi | Macro-F1 bỏ text | Sau khi xoá | Accuracy | Sau khi xoá | Đổi nhãn (caption bị đổi) | Dịch xác suất (caption bị đổi) |
|---|---|---|---|---|---|---|---|---|
| `distill_flat_a_s1` | source_tags | 193 | 0.537 | 0.535 | 0.767 | 0.752 | 0.228 | 0.147 |
| `distill_flat_a_s1` | all_tags | 313 | 0.537 | 0.535 | 0.767 | 0.754 | 0.176 | 0.121 |
| `distill_flat_a_s2` | source_tags | 193 | 0.520 | 0.502 | 0.727 | 0.718 | 0.285 | 0.211 |
| `distill_flat_a_s2` | all_tags | 313 | 0.520 | 0.506 | 0.727 | 0.721 | 0.224 | 0.185 |
| `distill_flat_a_s42` | source_tags | 193 | 0.507 | 0.494 | 0.779 | 0.765 | 0.275 | 0.107 |
| `distill_flat_a_s42` | all_tags | 313 | 0.507 | 0.491 | 0.779 | 0.762 | 0.201 | 0.084 |

Theo từng thẻ nguồn, xoá theo cách `source_tags`:

| Run | Thẻ | Số mẫu val | Nhãn thật | Dự đoán trước | Dự đoán sau | Accuracy trước | Accuracy sau |
|---|---|---|---|---|---|---|---|
| `distill_flat_a_s1` | `interpool` | 90 | not 18%, text 0%, image 2%, multi 80% | not 22%, text 0%, image 6%, multi 72% | not 41%, text 0%, image 3%, multi 56% | 0.700 | 0.556 |
| `distill_flat_a_s1` | `voz` | 103 | not 60%, text 0%, image 6%, multi 34% | not 39%, text 0%, image 46%, multi 16% | not 18%, text 0%, image 61%, multi 20% | 0.350 | 0.272 |
| `distill_flat_a_s2` | `interpool` | 90 | not 18%, text 0%, image 2%, multi 80% | not 27%, text 0%, image 3%, multi 70% | not 41%, text 0%, image 6%, multi 53% | 0.700 | 0.578 |
| `distill_flat_a_s2` | `voz` | 103 | not 60%, text 0%, image 6%, multi 34% | not 39%, text 0%, image 22%, multi 39% | not 28%, text 0%, image 6%, multi 66% | 0.476 | 0.456 |
| `distill_flat_a_s42` | `interpool` | 90 | not 18%, text 0%, image 2%, multi 80% | not 30%, text 0%, image 0%, multi 70% | not 47%, text 0%, image 0%, multi 53% | 0.678 | 0.556 |
| `distill_flat_a_s42` | `voz` | 103 | not 60%, text 0%, image 6%, multi 34% | not 85%, text 0%, image 0%, multi 15% | not 49%, text 0%, image 0%, multi 51% | 0.573 | 0.485 |

`interpool`, `voz` là thẻ của trang nguồn (`#interpool`, `#WMO`, `#voz_f33`). Model vẫn đã học với thẻ, nên đây chỉ đo mức phụ thuộc lúc suy luận; muốn biết model học được gì khi không có thẻ thì phải train lại.

- 193/1 413 caption val có thẻ. Xoá thẻ làm 26% dự đoán trên các caption này đổi nhãn (trung bình 3 seed), macro-F1 bỏ `text-sarcasm` giảm từ 0,521 xuống 0,511, accuracy từ 0,758 xuống 0,745.
- Xoá thêm mọi hashtag và chữ ký không làm giảm thêm đáng kể.
- Trên 90 bài `interpool`, accuracy giảm từ khoảng 0,69 xuống 0,56: model dựa vào thẻ này, và ở val điều đó có lợi vì 80% các bài này là `multi-sarcasm`.
- Hashtag trong train gắn chặt với nhãn: `#wmo`, `#worldmemeorganization` khoảng 65% `image-sarcasm` (toàn tập 4%), `#quytacungxutx` 100% `multi-sarcasm` (55 bài), `#redbull`, `#vtv`, `#70namdienbienphu` 100% `not-sarcasm` (30 bài mỗi thẻ). Nhiều thẻ trong số này không xuất hiện ở val, chi tiết ở `outputs/diagnostics/source_tags_train.csv`.

| Cấu hình | Số seed | Macro-F1 | Macro-F1 bỏ text | Weighted-F1 | Accuracy | F1 `not-sarcasm` | F1 `text-sarcasm` | F1 `image-sarcasm` | F1 `multi-sarcasm` |
|---|---|---|---|---|---|---|---|---|---|
| `distill_flat_a` | 3 | 0.391 ± 0.011 | 0.521 ± 0.015 | 0.760 ± 0.022 | 0.758 ± 0.027 | 0.657 ± 0.030 | 0.000 ± 0.000 | 0.077 ± 0.069 | 0.830 ± 0.022 |
| `distill_flat_a_notags` | 3 | 0.376 ± 0.014 | 0.501 ± 0.019 | 0.754 ± 0.004 | 0.759 ± 0.017 | 0.635 ± 0.014 | 0.000 ± 0.000 | 0.036 ± 0.062 | 0.832 ± 0.009 |

Train lại và đánh giá khi caption không có thẻ nguồn, hashtag và chữ ký (`distill_flat_a_notags`, 3 seed): macro-F1 bỏ `text-sarcasm` giảm từ 0,521 xuống 0,501 (cả 3 seed đều giảm, 1,4–3,4 điểm), accuracy không đổi (0,758 và 0,759). F1 `multi-sarcasm` gần như giữ nguyên; `not-sarcasm` giảm từ 0,657 xuống 0,635; `image-sarcasm` từ 0,077 xuống 0,036, chỉ có 28 mẫu val nên rất nhiễu. Chênh lệch khoảng bằng 1 độ lệch chuẩn giữa các seed. Thẻ có đóng góp nhưng không phải nguồn tín hiệu chính. Logo và chữ trong ảnh vẫn còn trong OCR nên nguồn chưa bị loại bỏ hoàn toàn.

### 5.7. E1: kNN retrieval

| Run | Chưa hiệu chỉnh prior | Prior, không kNN (λ = 1) | kNN tốt nhất | kNN, kiểm định chéo | Tham số | F1 `image`, λ = 1 | F1 `image`, kNN | sim láng giềng gần nhất | Tỉ lệ sim ≥ 0,9 | Trùng nhãn láng giềng |
|---|---|---|---|---|---|---|---|---|---|---|
| `distill_flat_a_s1` | 0.537 | 0.502 | 0.504 | 0.488 | k=50, τ=0.1, λ=0.9 | 0.040 | 0.044 | 0.790 | 0.013 | 0.616 |
| `distill_flat_a_s2` | 0.520 | 0.502 | 0.511 | 0.504 | k=5, τ=0.2, λ=0.5 | 0.085 | 0.057 | 0.604 | 0.001 | 0.632 |
| `distill_flat_a_s42` | 0.507 | 0.489 | 0.489 | 0.481 | k=20, τ=0.1, λ=0.7 | 0.000 | 0.000 | 0.832 | 0.035 | 0.621 |
| `distill_hier_b_s1` | 0.512 | 0.460 | 0.482 | 0.480 | k=5, τ=0.1, λ=0.5 | 0.000 | 0.000 | 0.722 | 0.003 | 0.630 |
| `distill_hier_b_s2` | 0.519 | 0.466 | 0.501 | 0.501 | k=5, τ=0.2, λ=0.5 | 0.000 | 0.062 | 0.725 | 0.003 | 0.636 |
| `distill_hier_b_s42` | 0.504 | 0.486 | 0.493 | 0.482 | k=10, τ=0.1, λ=0.7 | 0.000 | 0.000 | 0.808 | 0.017 | 0.616 |
| `hier_cascade_s1` | 0.499 | 0.477 | 0.488 | 0.480 | k=5, τ=0.05, λ=0.5 | 0.000 | 0.000 | 0.817 | 0.018 | 0.604 |
| `hier_cascade_s2` | 0.514 | 0.483 | 0.492 | 0.490 | k=5, τ=0.05, λ=0.5 | 0.000 | 0.000 | 0.802 | 0.019 | 0.633 |
| `hier_cascade_s42` | 0.507 | 0.472 | 0.484 | 0.480 | k=5, τ=0.1, λ=0.5 | 0.000 | 0.000 | 0.814 | 0.020 | 0.607 |

Không hiệu chỉnh prior ở cả model và kNN (`--no-prior`); cột "Prior, không kNN" khi đó là model gốc:

| Run | Chưa hiệu chỉnh prior | Prior, không kNN (λ = 1) | kNN tốt nhất | kNN, kiểm định chéo | Tham số | F1 `image`, λ = 1 | F1 `image`, kNN | sim láng giềng gần nhất | Tỉ lệ sim ≥ 0,9 | Trùng nhãn láng giềng |
|---|---|---|---|---|---|---|---|---|---|---|
| `distill_flat_a_s1` | 0.537 | 0.537 | 0.540 | 0.529 | k=20, τ=0.05, λ=0.9 | 0.102 | 0.106 | 0.790 | 0.013 | 0.616 |
| `distill_flat_a_s2` | 0.520 | 0.520 | 0.528 | 0.500 | k=50, τ=0.2, λ=0.5 | 0.130 | 0.120 | 0.604 | 0.001 | 0.632 |
| `distill_flat_a_s42` | 0.507 | 0.507 | 0.509 | 0.503 | k=50, τ=0.05, λ=0.9 | 0.000 | 0.000 | 0.832 | 0.035 | 0.621 |
| `distill_hier_b_s1` | 0.512 | 0.512 | 0.539 | 0.516 | k=10, τ=0.05, λ=0.9 | 0.069 | 0.133 | 0.722 | 0.003 | 0.630 |
| `distill_hier_b_s2` | 0.519 | 0.519 | 0.534 | 0.512 | k=50, τ=0.05, λ=0.5 | 0.105 | 0.118 | 0.725 | 0.003 | 0.636 |
| `distill_hier_b_s42` | 0.504 | 0.504 | 0.507 | 0.505 | k=10, τ=0.1, λ=0.9 | 0.000 | 0.000 | 0.808 | 0.017 | 0.616 |
| `hier_cascade_s1` | 0.499 | 0.499 | 0.504 | 0.499 | k=10, τ=0.2, λ=0.5 | 0.000 | 0.000 | 0.817 | 0.018 | 0.604 |
| `hier_cascade_s2` | 0.514 | 0.514 | 0.517 | 0.493 | k=50, τ=0.1, λ=0.5 | 0.053 | 0.065 | 0.802 | 0.019 | 0.633 |
| `hier_cascade_s42` | 0.507 | 0.507 | 0.509 | 0.507 | k=10, τ=0.05, λ=0.9 | 0.000 | 0.000 | 0.814 | 0.020 | 0.607 |

Trên val, theo tiêu chí macro-F1 bỏ `text-sarcasm`. "Kiểm định chéo" chọn tham số trên phần còn lại của val, nên công bằng hơn "kNN tốt nhất". Mốc λ = 1 có hiệu chỉnh prior như thiết kế.

**Không giữ E1.** So với λ = 1, kiểm định chéo: `distill_flat_a` -0,6 điểm, `distill_hier_b` +1,7, `hier_cascade` +0,6. Hai hàng sau vượt mức lệch giữa các seed, nhưng cả ba hàng vẫn thấp hơn cùng model chưa hiệu chỉnh prior khoảng 2,3–3,0 điểm (0,491/0,488/0,484 so với 0,521/0,512/0,507), và phần tăng chủ yếu bù lại mức giảm do hiệu chỉnh prior. F1 `image-sarcasm` không cải thiện.

**Bản không hiệu chỉnh prior (`--no-prior`), so trực tiếp với model gốc:** kiểm định chéo thấp hơn model gốc ở cả 3 hàng: `distill_flat_a` -1,1 điểm, `distill_hier_b` -0,05, `hier_cascade` -0,7. Điểm dò trên chính val có tăng 0,3–1,5 điểm nhưng mất khi chọn tham số ngoài mẫu, tức là phần tăng đó là do dò quá khớp với val. F1 `image-sarcasm` không tăng ở seed nào. Kết luận không giữ E1 vẫn đúng với cả hai cách.

Giả định "meme lặp template" không được ủng hộ: chỉ 0,1–3,5% mẫu val có láng giềng gần nhất với sim ≥ 0,9, và nhãn láng giềng gần nhất chỉ đúng 60–64%, thấp hơn luôn đoán `multi-sarcasm` (66,7%).

## 6. Giai đoạn 1: thử teacher

### prompt-dev (`pilot_dev`, 160 mẫu cân bằng)

| Lần chạy | Teacher | Prompt | Số mẫu | Accuracy | Macro-F1 | F1 `not-sarcasm` | F1 `text-sarcasm` | F1 `image-sarcasm` | F1 `multi-sarcasm` | Parse lỗi | Giây/lần sinh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `pilot_dev_v1_4bit` | Qwen/Qwen2.5-VL-7B-Instruct | v1 | 160 | 0.269 | 0.190 | 0.393 | 0.115 | 0.000 | 0.250 | 0.0% | 1.6 |
| `pilot_dev_v2_4bit` | Qwen/Qwen2.5-VL-7B-Instruct | v2 | 160 | 0.256 | 0.161 | 0.384 | 0.047 | 0.000 | 0.215 | 0.6% | 1.7 |
| `pilot_dev_v3_4bit` | Qwen/Qwen2.5-VL-7B-Instruct | v3 | 160 | 0.263 | 0.160 | 0.405 | 0.048 | 0.000 | 0.188 | 0.0% | 1.8 |
| `pilot_dev_v4_4bit` | Qwen/Qwen2.5-VL-7B-Instruct | v4 | 160 | 0.256 | 0.165 | 0.369 | 0.000 | 0.000 | 0.289 | 0.6% | 2.2 |
| `pilot_dev_v5_4bit` | Qwen/Qwen2.5-VL-7B-Instruct | v5 | 160 | 0.256 | 0.180 | 0.347 | 0.047 | 0.000 | 0.326 | 2.5% | 3.4 |
| `pilot_dev_v5_gemini` | gemini-pro-latest | v5 | 160 | 0.487 | 0.480 | 0.442 | 0.411 | 0.637 | 0.430 | 0.0% | 12.8 |
| `pilot_dev_v6_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v6 | 160 | 0.438 | 0.434 | 0.390 | 0.426 | 0.565 | 0.353 | 2.5% | 1.7 |
| `pilot_dev_v6nodesc_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v6_nodesc | 160 | 0.487 | 0.474 | 0.463 | 0.375 | 0.660 | 0.400 | 1.9% | 1.9 |

### pilot-eval (`pilot_eval`, 120 mẫu)

### `natural_dev` (500 mẫu train ngẫu nhiên, ngoài kế hoạch gốc)

| Lần chạy | Teacher | Prompt | Số mẫu | Accuracy | Macro-F1 | F1 `not-sarcasm` | F1 `text-sarcasm` | F1 `image-sarcasm` | F1 `multi-sarcasm` | Parse lỗi | Giây/lần sinh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `natural_dev_v6_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v6 | 500 | 0.500 | 0.318 | 0.677 | 0.000 | 0.096 | 0.500 | 2.6% | 2.3 |
| `natural_dev_v6nodesc_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v6_nodesc | 500 | 0.496 | 0.317 | 0.678 | 0.000 | 0.095 | 0.494 | 1.6% | 1.7 |
| `natural_dev_v7_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v7 | 500 | 0.588 | 0.346 | 0.677 | 0.000 | 0.098 | 0.609 | 2.0% | 1.8 |
| `natural_dev_v7_qwen3-vl-8b` | Qwen/Qwen3-VL-8B-Instruct | v7 | 500 | 0.468 | 0.204 | 0.208 | 0.000 | 0.000 | 0.606 | 1.8% | 8.5 |
| `natural_dev_v8_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v8 | 500 | 0.568 | 0.326 | 0.655 | 0.000 | 0.047 | 0.602 | 2.6% | 2.4 |
| `natural_dev_v9_qwen3-vl-8b` | Qwen/Qwen3-VL-8B-Instruct | v9 | 500 | 0.506 | 0.231 | 0.302 | 0.000 | 0.000 | 0.623 | 0.2% | 7.7 |

### val (public test, 1 413 mẫu)

| Lần chạy | Teacher | Prompt | Số mẫu | Accuracy | Macro-F1 | F1 `not-sarcasm` | F1 `text-sarcasm` | F1 `image-sarcasm` | F1 `multi-sarcasm` | Parse lỗi | Giây/lần sinh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `val_v8_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v8 | 1413 | 0.589 | 0.339 | 0.561 | 0.000 | 0.097 | 0.699 | 3.4% | 2.4 |
| `val_zero_shot_v6_gemini-3.5-flash-lite` | gemini-3.5-flash-lite | v6 | 1413 | 0.420 | 0.286 | 0.565 | 0.000 | 0.075 | 0.501 | 3.0% | 2.2 |

### Đối chiếu với mốc G1 trên val

| Lần chạy | F1 `image` | Mốc | F1 `multi` | Mốc | Parse lỗi | G1 |
|---|---|---|---|---|---|---|
| `val_v8_gemini-3.5-flash-lite` | 0.097 | 0.077 | 0.699 | 0.853 | 3.4% | đạt |
| `val_zero_shot_v6_gemini-3.5-flash-lite` | 0.075 | 0.077 | 0.501 | 0.853 | 3.0% | không đạt |

**Mức độ tin cậy của các lần chạy.** Prompt v1–v5 chạy Qwen2.5-VL-7B lượng tử 4-bit do GPU 12 GB không chứa được bản bf16. `pilot_eval` chưa dùng. Val đã dùng hai lần: v6, rồi v8. v7 được viết sau khi xem kết quả v6 trên val, nên kết quả v8 trên val có thể hơi lạc quan. `natural_dev` được thêm vì `pilot_dev` cân bằng lớp không phản ánh phân bố thật. Từ v5 prompt có 4 ví dụ cố định lấy từ train, nên các lần chạy này không còn là zero-shot hoàn toàn dù tên thư mục có `zero_shot`.

**Đổi model.** `gemini-2.5-flash-lite` trả về 404 (đóng với người dùng mới) nên dùng `gemini-3.5-flash-lite`. `gemini-pro-latest` khi chạy trỏ tới `gemini-3.1-pro-preview`.

**Bỏ mô tả VLM.** So cặp với v6 không thấy khác biệt đo được: trên `natural_dev` macro-F1 0,317 so với 0,318 (khoảng tin cậy 95% của chênh lệch, tính khi coi mẫu không đọc được là một nhóm riêng, từ -0,025 đến 0,019), ít hơn 14% token đầu vào. v7 không gửi mô tả ảnh.

**Qwen3-VL-8B 4-bit, v7.** Dự đoán `multi-sarcasm` cho 458/491 mẫu đọc được, thấp hơn việc luôn đoán `not-sarcasm` (56%). So với Flash Lite, khoảng tin cậy 95% của chênh lệch macro-F1 là -0,178 đến -0,109. Luật "khi phân vân thì chọn multi" của v7 có thể góp phần vào kết quả này. Chạy ở batch 1 vì batch 2 hết bộ nhớ khi có 4,2 GB đang bị chiếm trên GPU.

**Bỏ luật "khi phân vân thì chọn multi" (v8).** Trên `natural_dev`, phân bố dự đoán gần như không đổi so với v7: accuracy 56,8% so với 58,8%, macro-F1 0,326 so với 0,346. Câu nói `multi-sarcasm` phổ biến hơn vẫn được giữ. Trên val, 196/942 mẫu `multi-sarcasm` bị đoán thành `image-sarcasm`; accuracy 58,9%, thấp hơn luôn đoán `multi-sarcasm` (66,7%).

**Prompt v9 (định nghĩa chính thức, quy tắc suy ra từ dữ liệu, 7 ví dụ có mô tả ảnh) với Qwen3-VL-8B 4-bit trên `natural_dev`:** accuracy 50,6% (v7: 46,8%), macro-F1 bỏ `text-sarcasm` 0,308 (v7: 0,272), parse lỗi 0,2%. Vẫn đoán `multi-sarcasm` cho 446/499 mẫu, 230/280 mẫu `not-sarcasm` bị đoán `multi`, và `image-sarcasm` 0/15. Vẫn thấp hơn luôn đoán `not-sarcasm` (56%) và thấp xa Flash Lite v8 (0,435). Cờ châm biếm có giảm (caption: 87% xuống 75% trên mẫu `not-sarcasm`) nhưng vẫn bật cho phần lớn bài không châm biếm. Prompt tốt hơn không khắc phục được thiên lệch của model này.

**Chi phí (ước tính, giá lấy từ nguồn tổng hợp, chưa đối chiếu giá chính thức).** Flash Lite khoảng 0,0014 USD mỗi lần sinh: khoảng 10 USD cho lô 1–3 với k = 1, khoảng 30 USD với k theo kế hoạch. Gemini Pro khoảng 35 USD cho val và khoảng 500 USD cho các lô.

### E2: few-shot

## 7. Giai đoạn 2: sinh reasoning

**Thử giải thích khi đã cho nhãn đúng (Qwen3-VL-8B 4-bit, v8, 100 mẫu train, ngoài `pilot_eval`), 8 giây/mẫu, chưa sinh cả tập.** 98/100 đọc được; 92/100 kết luận đúng nhãn đã cho, riêng `text-sarcasm` 16/20. Cờ text/ảnh khớp nhãn ở 95/100 mẫu, tức gần như chép lại nhãn gốc, nên cột (c) sẽ gần trùng cột (b).

Đọc thử 13 mẫu, đối chiếu ảnh ở 4 mẫu (chưa phải chấm rubric): mô tả bề mặt và lý do cho mẫu `not-sarcasm` thường đọc được; phần giải thích ý mỉa mai yếu. Lỗi gặp: nhận nhầm ảnh miệng rùa biển thành cá mập (1244); bịa nghĩa "số 13 xui xẻo" trong khi câu đùa là chồng được nhận lại máy cũ của vợ (4032); bỏ sót chơi chữ Nissan/Mitsubishi trong meme (2559); lý do mâu thuẫn với nhãn được giao (2259, 3832). File chấm: `outputs/cot_distill/label_given_review.csv`.

**Quyết định (2026-11-03): dừng M1, M2, M3.** Teacher miễn phí (Qwen) không viết được lý do đáng tin, teacher trả phí không còn ngân sách. Phương án người viết giải thích cũng bị loại: nhóm không suy ra được quy tắc gán nhãn của bộ dữ liệu, bộ dữ liệu không công bố hướng dẫn gán nhãn, và `docs/labeling_guidelines.md` được dựng từ vài mẫu nên không khái quát được. Lý do viết cho nhãn mà người viết không hiểu sẽ chỉ là hợp lý hoá. Sheet đã tạo (`outputs/human_explanations/`) không được dùng.

## 8. Giai đoạn 3: phân nhóm và rationalization

## 9. Giai đoạn 4: chấm theo rubric

## 10. Giai đoạn 5: đóng gói dữ liệu

## 11. Giai đoạn 6: ma trận thí nghiệm

### 11.1. Macro-F1 trên val

| Cấu trúc đầu ra | (a) CE | (b) + nhãn phụ từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) đầy đủ |
|---|---|---|---|---|
| Phẳng 4 lớp | 0.391 ± 0.011 | 0.389 ± 0.014 |  |  |
| Phân cấp mềm | 0.379 ± 0.003 | 0.384 ± 0.006 |  |  |
| Subclass |  |  |  |  |
| Phân cấp nối tiếp | 0.380 ± 0.006 |  |  |  |

### 11.2. Macro-F1 trên val, có hiệu chỉnh prior

| Cấu trúc đầu ra | (a) CE | (b) + nhãn phụ từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) đầy đủ |
|---|---|---|---|---|
| Phẳng 4 lớp | 0.373 ± 0.006 | 0.366 ± 0.008 |  |  |
| Phân cấp mềm | 0.354 ± 0.013 | 0.353 ± 0.010 |  |  |
| Subclass |  |  |  |  |
| Phân cấp nối tiếp | 0.358 ± 0.004 |  |  |  |

### 11.3. Macro-F1 trên private test

| Cấu trúc đầu ra | (a) CE | (b) + nhãn phụ từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) đầy đủ |
|---|---|---|---|---|
| Phẳng 4 lớp |  |  |  |  |
| Phân cấp mềm |  |  |  |  |
| Subclass |  |  |  |  |
| Phân cấp nối tiếp |  |  |  |  |

### 11.4. Macro-F1 trên private test, có hiệu chỉnh prior

| Cấu trúc đầu ra | (a) CE | (b) + nhãn phụ từ nhãn gốc | (c) + nhãn phụ từ teacher | (d) đầy đủ |
|---|---|---|---|---|
| Phẳng 4 lớp |  |  |  |  |
| Phân cấp mềm |  |  |  |  |
| Subclass |  |  |  |  |
| Phân cấp nối tiếp |  |  |  |  |

### 11.5. Ablation: bỏ reasoning nhóm 3 (val)

| Cấu hình | Số seed | Macro-F1 | Macro-F1 bỏ text | Weighted-F1 | Accuracy | F1 `not-sarcasm` | F1 `text-sarcasm` | F1 `image-sarcasm` | F1 `multi-sarcasm` |
|---|---|---|---|---|---|---|---|---|---|
| `distill_flat_d_nogroup3` |  |  |  |  |  |  |  |  |  |
| `distill_hier_d_nogroup3` |  |  |  |  |  |  |  |  |  |
| `distill_sub_d_nogroup3` |  |  |  |  |  |  |  |  |  |

### 11.6. D1 sau distillation

### 11.7. E1 trên từng hàng

## 12. Giai đoạn 7: các nhánh tuỳ chọn

### B3: reasoning làm đầu vào

### B2: 3 model riêng + stacking

| Cách | Out-of-fold trên train | Val | Val, hiệu chỉnh prior | F1 `image`, val |
|---|---|---|---|---|
| Model `text` riêng | 0.496 | 0.504 | 0.479 | 0.000 |
| Model `image` riêng | 0.435 | 0.467 | 0.476 | 0.000 |
| Model `multi` riêng | 0.526 | 0.507 | 0.489 | 0.000 |
| Trung bình xác suất 3 model |  | 0.500 | 0.484 | 0.000 |
| Stacking, tốt nhất trên val (0.01, none, False) |  | 0.505 |  | 0.000 |
| Stacking, kiểm định chéo trong val |  | 0.503 |  | 0.000 |

1 seed (42), macro-F1 bỏ `text-sarcasm` trên val. Model đa phương thức của B2 cùng cấu hình và seed với `distill_flat_a_s42` và cho đúng cùng điểm (0,507), nên các con số so được trực tiếp với seed đó, không phải với trung bình 3 seed (0,521).

**Không giữ B2.** Stacking 0,504 (kiểm định chéo 0,504), trung bình xác suất 0,501, đều không hơn model đa phương thức một mình (0,507). Classifier gộp gần như chỉ dựa vào model đa phương thức. F1 `image-sarcasm` bằng 0 ở cả 3 model riêng và ở bản gộp: model chỉ-ảnh cũng không học được lớp này.

**Text-only cho D1:** model chỉ caption đạt 0,504, gần bằng model đa phương thức (0,507) và hơn model chỉ-ảnh (0,467). Ảnh, OCR và mô tả gần như không thêm gì vào điểm tổng, khớp với kết quả đổi ảnh của D1. Cần nhiều seed hơn để tách được chênh lệch 0,3 điểm này khỏi nhiễu.

### E3: pretrain trên dữ liệu ngoài

## 13. Trả lời các câu hỏi thí nghiệm

| So sánh | Câu hỏi | Kết quả |
|---|---|---|
| Phẳng (a) với phân cấp mềm (a) | Chỉ tách tầng S/C (chưa có teacher) đã giúp chưa? Phần lớn lợi ích dự kiến đến từ việc hiệu chỉnh label shift ở tầng S. |  |
| Phân cấp mềm (a) với nối tiếp (a) | Phân cấp mềm có tránh được lỗi lan truyền của Vu et al. không? |  |
| Cột (b) với cột (c) | Nhãn "text/ảnh đứng riêng" của teacher có thêm thông tin so với nhãn suy từ nhãn gốc không? |  |
| Cột (c) với cột (d) | Căn chỉnh với embedding reasoning có thêm gì ngoài nhãn phụ không? |  |
| **M1 với M2** | Đầu ra phân cấp có giúp student tận dụng reasoning tốt hơn không? Nên xem F1 từng lớp, nhất là `image-sarcasm`, và confusion giữa `multi` với `text`/`image`. |  |
| **M3 với M1, M2** | Đưa T/I của teacher vào không gian nhãn có tốt hơn dùng làm loss phụ (M1) hay đặc trưng trung gian (M2) không? Xem recall của `multi` trên val theo từng trường hợp 1–4, nhất là trường hợp 4. Trường hợp lấy từ T/I trong lần chạy zero-shot trên val của teacher. Các nhãn này chưa qua rubric nên chỉ để phân tích, và cần chấm một phần mẫu để ước lượng độ tin cậy. |  |
| **E1 với cùng model không retrieval** | Retrieval có thêm thông tin không? Xem cả `λ` được chọn và phân bố `sim`. |  |
| **M1, M2, M3 với B2** | Train end-to-end có hơn gộp muộn nhiều model không? |  |
| **E3 với cùng cấu hình không pretrain** | Dữ liệu ngoài giúp hay gây chuyển giao âm? |  |
| **E2 với zero-shot** | Few-shot có làm teacher tốt hơn và giảm nhóm 3 không? Reasoning của nó có đạt rubric nhiều hơn không? |  |
| **D1 trước và sau distillation** | Reasoning có làm model dùng ảnh nhiều hơn và bớt phụ thuộc tín hiệu bề mặt không? |  |

## 14. Kết quả phụ của teacher

## 15. Ví dụ định tính

## 16. Hạn chế

- Hướng dẫn gán nhãn chính thức (`docs/annotation_guideline.md`) chỉ định nghĩa có/không châm biếm; ranh giới giữa `text`, `image` và `multi` chỉ có 9 ví dụ. 5 ví dụ có trong bộ dữ liệu, 3 trong số đó mang nhãn khác với slide (train 8517, train 493, private test 545).
- Val chỉ có 1 mẫu `text-sarcasm` và 28 mẫu `image-sarcasm`, nên F1 các lớp này rất nhạy với vài mẫu.
