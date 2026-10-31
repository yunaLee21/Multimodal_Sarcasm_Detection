# Kế hoạch thí nghiệm: CoT distillation và các hướng liên quan

Thiết kế phương pháp và ý nghĩa các mã M1–M3, B1–B3, E1–E3, D1: `docs/experiment_design.md` (mục 1.1). Tài liệu này chỉ nói **chạy gì, theo thứ tự nào, ra file gì, khi nào dừng**.

Nguyên tắc:
- Mọi bước chạy trên **GPU server**, chỉ bằng script `.py` (không notebook), trong `tmux`. Cách cài đặt, bố cục dữ liệu và lệnh xem `CODEBASE.md` mục 7. Máy local chỉ dùng để viết code và chạy `pytest`.
- Cache sinh ra (`rationale_dataset.json`...) được upload lên Kaggle dataset để các thành viên chạy Kaggle dùng chung (`CODEBASE.md` mục 7.6).
- Mỗi giai đoạn kết thúc bằng một **cổng quyết định** (G0–G5). Không qua cổng thì không làm giai đoạn sau, mà quay lại sửa.
- Dùng `data.split=official` xuyên suốt: train = train, val = public test, test = private test. Private test chỉ xem ở bước đánh giá cuối của giai đoạn 6, và ở giai đoạn 7 cho các nhánh tuỳ chọn, không dùng để chọn model.

---

## Tổng quan

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

---

## Giai đoạn 0: nền móng, không cần teacher

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

---

## Giai đoạn 1: thử teacher

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

---

## Giai đoạn 2: sinh reasoning cho tập train

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

---

## Giai đoạn 3: phân nhóm, rationalization, xuất file chấm

**Code.** `scripts/build_rationale_dataset.py`, gồm 3 lệnh con:
- `group`: tính `n_correct / k`, gán nhóm 1/2/3 (`experiment_design.md` mục 4.1), chọn reasoning đại diện.
- `rationalize`: liệt kê id nhóm 3, gọi `generate_rationales.py --with-label`.
- `export`: xuất `review/<lô>.csv` với các cột `id`, `label`, `caption`, `ocr`, `image_path`, `rationale`, `group`, `rationalized`, và 4 cột rubric để trống.

**Chạy.** Chạy rationalization cho nhóm 3 trên server. Số lần gọi bằng số mẫu nhóm 3, với k = 1.

**Đầu ra.** File CSV để chấm, theo lô, và thống kê số mẫu từng nhóm theo lớp. Nên báo cáo thống kê này ở mục 6.5 của tài liệu thiết kế.

---

## Giai đoạn 4: người kiểm tra

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

---

## Giai đoạn 5: đóng gói dữ liệu

1. `build_rationale_dataset.py merge`: gộp kết quả chấm thành `rationale_dataset.json` theo định dạng ở `experiment_design.md` mục 4.4.
2. Mã hoá reasoning bằng PhoBERT đóng băng thành `rationale_embeddings.npy` cùng danh sách id. Tính trước một lần để không phải chạy PhoBERT trên reasoning mỗi epoch.
3. Lưu ở `data/processed/rationales/` trên server. Thêm `data.rationale.dir` / `file` vào `base.yaml`, theo cách đang làm với `image_text`. Đồng thời upload thành **Kaggle dataset** riêng, ví dụ `vimmsd-rationales`, phiên bản ứng với `prompt_version` / `rubric_version`, để nhóm dùng lại được trên Kaggle.
4. Code: `data/dataset.py` đọc file, thêm `text_sarcastic`, `image_sarcastic`, `e_rationale` và mask "có reasoning" vào batch. Có test cho mẫu không có reasoning.
5. Thống kê số mẫu mỗi subclass `multi_*`. Subclass dưới khoảng 30 mẫu đạt rubric thì gộp với subclass gần nhất (mục 5.4), ghi quyết định lại trước khi train M3.

---

## Giai đoạn 6: train ma trận thí nghiệm

**Code M3** (local, có test, làm trước khi chạy ma trận): `SubclassHead`, `λ_sub`, nhánh `subclass` của `compute_prior_shift`, config (`experiment_design.md` mục 8). Test gồm: xác suất gộp đúng bằng tổng 4 subclass và cộng lại bằng 1, mẫu `multi` thiếu nhãn teacher chỉ chịu loss mức thô, hiệu chỉnh prior giữ nguyên tỉ lệ giữa các subclass.

**Configs.** Gồm `distill_flat_{c,d}.yaml`, `distill_hier_{c,d}.yaml`, `distill_sub_{c,d}.yaml` và `distill_{flat,hier,sub}_d_nogroup3.yaml`. Kết hợp với cột (a)/(b) và cascade ở giai đoạn 0 là đủ ma trận ở `experiment_design.md` mục 6.2.

**Chọn hyperparameter trên val.** Grid nhỏ cho `λ_axis` ∈ {0,1; 0,3; 1} và `λ_rat` ∈ {0,1; 0,3; 1}, chỉ chạy trên cột (d), 1 seed. M3 dò riêng `λ_sub` ∈ {0,1; 0,3; 1} thay cho `λ_axis`. Sau đó cố định giá trị tốt nhất cho mọi ô.

**Chạy chính thức.** Mỗi ô 3 seed, tiêu chí chọn checkpoint giống nhau cho mọi ô. Không chọn theo macro-F1 4 lớp trên val, vì `text-sarcasm` chỉ có 1 mẫu trong val. Nên dùng weighted-F1, hoặc macro-F1 trên 3 lớp còn lại. Chốt tiêu chí trước khi chạy và ghi vào `reports/results.md`.

**Lặp lại trên val.** Chạy D1 cho checkpoint tốt nhất của M1, M2, M3 để so với baseline. Chạy E1 và chốt `k`, `τ`, `λ` cho từng hàng trước bước đánh giá cuối.

**Đánh giá cuối.** Chỉ sau khi mọi ô đã chọn xong checkpoint mới chạy `scripts/evaluate.py --split test` (private test) cho toàn bộ ma trận một lần, kèm phiên bản có E1 với siêu tham số đã chốt.

**Cổng G3.**
- Mọi ô đã đủ 3 seed. Báo cáo trung bình ± độ lệch chuẩn.
- M1 hoặc M2 hơn ô (b) cùng hàng, hoặc M3 hơn ô (b) của hàng phẳng, **quá 1 độ lệch chuẩn** thì mới kết luận reasoning có tác dụng. Nếu không, báo cáo đó là kết quả âm tính, và vẫn giữ phân tích ở mục 6.3 của tài liệu thiết kế.

---

## Giai đoạn 7 (tuỳ chọn): các nhánh mở rộng

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

---

## Giai đoạn 8: phân tích và báo cáo

**Cổng G4**: báo cáo đủ các mục sau.
1. Bảng ma trận đầy đủ trên private test (cả có và không có hiệu chỉnh prior), so với mốc 0,41 của Vu et al., ghi rõ loại F1.
2. Trả lời từng câu hỏi ở `experiment_design.md` mục 6.3.
3. Kết quả của teacher: zero-shot trên val, độ chính xác trên train theo lớp, phân bố 4 trường hợp của `multi`.
4. Thống kê lọc và chấm: tỉ lệ từng nhóm, tỉ lệ đạt rubric, κ.
5. 5–10 ví dụ định tính, gồm mẫu M2 hoặc M3 đúng mà baseline sai và ngược lại, kèm reasoning của teacher.
6. Kết quả D1 trước và sau distillation, E1 so với không retrieval, E2 so với zero-shot, và các nhánh tuỳ chọn đã chạy, kể cả nhánh dừng giữa chừng.
7. Hạn chế: số mẫu `text-sarcasm`, độ nhạy theo prior, chi phí.

Kết quả ghi vào `reports/results.md`. Bản tổng hợp riêng đặt ở `reports/experiment_report.md`.

---

## Theo dõi tiến độ

| # | Việc | Phụ thuộc | Trạng thái |
|---|---|---|---|
| S.1 | Cài server (`setup_env.sh`), tải data + cache có sẵn (`prepare_data.sh`), chạy plan `smoke` | – | đã có script, chưa chạy trên server |
| S.2 | `run_experiments.py`, `summarize_results.py`, `preprocess_text.py` | – | xong (có test), chưa chạy trên server |
| 0.1 | Head phân cấp + loss + test | – | có code + test |
| 0.2 | Nhãn phụ từ nhãn gốc + hiệu chỉnh prior | – | có code + test |
| 0.3 | Chạy plan `phase4_baselines` (mốc), rồi (a)/(b) phẳng, phân cấp, cascade × 3 seed | S.1, S.2, 0.1, 0.2 | chưa làm |
| 0.4 | D1: `diagnose_shortcuts.py` + test, chạy trên baseline | 0.3 | có code + test, chưa chạy |
| 0.5 | E1: `knn_eval.py` + test, chạy trên baseline | 0.3 | chưa có code |
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
