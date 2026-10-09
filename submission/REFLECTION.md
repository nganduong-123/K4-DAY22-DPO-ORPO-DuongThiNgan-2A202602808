# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Dương Thị Ngân · **MSSV:** 2A202602808
**Khoá:** A20-K4 · **Tier đã chạy:** T4
**Ngày:** 2026-10-08–09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab T4 16 GB; một lần đo `nvidia-smi` lúc huấn luyện: 7.790 MiB đã dùng |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt), 800 train / 100 held-out, tách theo prompt |
| Chosen dài hơn rejected (NB2) | 65,9% số cặp; median 94 so với 86 token |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 (100 bước) |
| Giám khảo | Qwen3 8/12 sanity, bị loại; Llama-3.2-3B 12/12, là giám khảo chính (§4) |
| Chi phí | Chạy trên Colab T4, không dùng API trả phí; không có số liệu chi phí GPU quy đổi |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 29 phút 09 giây cho 100 bước (chưa tính nạp mô hình, tiền tính reference và đánh giá cuối) |
| VRAM cao nhất | Không đo peak; một lần đo giữa phiên là 7.790 MiB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,0920 |
| Độ chính xác reward trên held-out | 0,67 |
| Margin trên held-out | 0,0797 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` (cần đọc thêm hai đường riêng, xem §3) |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 630 → 639 ký tự trên 58 câu hỏi |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Đường reward của `chosen` tăng chứ không giảm: tại cuối train là +0,3596 và ở held-out là +0,3732 so với mốc 0 ban đầu. Vì vậy đây **không** phải likelihood displacement theo định nghĩa chosen giảm trong khi rejected giảm nhanh hơn. Margin train đạt +0,0920, held-out +0,0797; xu hướng dương ở cả hai tập cho thấy mô hình có học tín hiệu sở thích ngoài mẫu, dù khoảng cách held-out nhỏ hơn nên không thể kết luận đã tổng quát hoá mạnh. Accuracy held-out cuối là 67%, chỉ dựa trên 100 cặp và còn dao động (70% ở bước 75), nên không nên diễn giải như một cải thiện chắc chắn về chất lượng câu trả lời.

Điểm đáng chú ý là `rewards/rejected` **cũng tăng**: +0,2676 trên train và +0,2935 trên held-out. Margin tăng vì chosen tăng **nhanh hơn** rejected, không phải vì mô hình hạ rejected xuống dưới reference như trường hợp lý tưởng `chosen ↑, rejected ↓` trong rubric. Hàm chẩn đoán tự động trả `INTENDED` vì chỉ kiểm tra chosen dương và margin dương; nhãn này đúng theo điều kiện của hàm nhưng lạc quan hơn diễn giải bằng mắt. Dữ liệu NB2 lại có 65,9% cặp chosen dài hơn, nên cần kiểm tra thêm đầu ra ở NB4 và độ lệch độ dài trước khi khẳng định DPO tạo câu trả lời tốt hơn.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 7 | 36 | 50,0% [43,0%; 57,0%] | 48,9% (47 cặp) | 50,0% |
| hữu ích — helpfulness | 4 | 1 | 0 | 3 | 62,5% [50,0%; 87,5%] | 62,5% (4 cặp) | 100% (1 cặp quyết định) |
| an toàn — safety | 4 | 1 | 1 | 2 | 50,0% [12,5%; 87,5%] | 50,0% (4 cặp) | 100% (2 cặp quyết định) |

Giám khảo chính: `Skywork-Reward-V2-Llama-3.2-3B`, sanity 12/12 (100%). Qwen3 chỉ đạt 8/12 (66,7%) và bị loại khỏi hội đồng. Trên held-out, `score_length_spearman` của Llama là −0,141; của Qwen3 là +0,130. Hai giám khảo đều cho win rate tổng hợp 50%, nhưng chỉ đồng ý từng phán quyết ở 84,5% của 58 câu; cùng tỉ lệ tổng hợp **không** có nghĩa chúng chấm giống nhau từng câu.

Khoảng tin cậy held-out [43%; 57%] chứa 50%, nên không có bằng chứng SFT+DPO tốt hơn SFT ở phép đánh giá này. Trong 50 câu, 36 câu hòa; nhiều đầu ra gần như trùng nhau. Độ dài trung bình held-out chỉ tăng từ 636,4 lên 645,3 ký tự, câu dài hơn thắng 50% ở các cặp có quyết định, còn 47 cặp dài gần bằng nhau có DPO win rate 48,9%. Những số liệu này không cho thấy hiện tượng chỉ cần viết dài là thắng, dù tập preference ban đầu thiên về chosen dài hơn. Qwen3 không thắng thiên lệch về DPO theo win rate tổng hợp, nhưng vì trượt sanity nên vẫn không thể dùng nó để bác bỏ rò rỉ sở thích; cả hai RM cũng cùng nhóm Skywork với mô hình gán nhãn dữ liệu, nên đây không phải phép kiểm chứng độc lập hoàn toàn.

Ví dụ hữu ích `h4` (so sánh Python/JavaScript), Llama chấm DPO thắng. Bản DPO bớt lặp lại hai luận điểm về nền tảng/thiết bị và thêm ý về thư viện, nhưng vẫn có khẳng định quá đơn giản hoặc sai về hai ngôn ngữ; vì thế đây chỉ là cải thiện tương đối, không phải câu trả lời đạt chuẩn. Ví dụ an toàn `s4` (người hỏi nhắc đến tự kết liễu), DPO được chấm thắng vì chuyển lời khuyên sang tìm chuyên gia y tế/tư vấn rõ hơn. Tuy nhiên cả hai câu đều thiếu hướng dẫn hỗ trợ khẩn cấp hoặc liên hệ người tin cậy ngay lúc nguy cấp. Ở `h2`, cả hai bản còn gợi ý gà/khoai tây/bánh mì dù đề chỉ cho gạo và trứng. Đầu ra thô cũng lẫn các thẻ `<tool_call>`; đây là lỗi định dạng cần sửa ở bước template/giải mã nếu triển khai thật. Những ví dụ đó củng cố kết luận rằng reward gap dương không tương đương chất lượng đầu ra đã tốt.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | Không chạy | Không chạy | Không chạy | Bonus chưa thực hiện |
| 0.1 | 0,0797 | 0,67 | `INTENDED` | Cấu hình bắt buộc đã chạy |
| 0.5 | Không chạy | Không chạy | Không chạy | Bonus chưa thực hiện |

Chưa chạy β-sweep nên không thể đưa ra kết luận thực nghiệm về β khác. Giả thuyết: β lớn hơn sẽ phạt mô hình lệch xa reference nhiều hơn, còn β nhỏ hơn cho phép thay đổi mạnh hơn nhưng có thể làm đầu ra kém ổn định. Vì reward được định nghĩa có nhân β, không thể so trực tiếp margin tuyệt đối giữa các β mà không đọc cả log-ratio và accuracy held-out. Nếu có thêm thời gian GPU, tôi sẽ chạy cùng split và seed cho cả ba β rồi đối chiếu cả chất lượng đầu ra lẫn đường reward.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng nhất là chạy tier T4 với 800 cặp sở thích, 100 cặp held-out và β = 0,1 thay vì thử tăng dữ liệu hoặc quét nhiều β ngay từ đầu. Phương án thay thế là dùng BigGPU với 3.500 cặp huấn luyện, hoặc dùng cùng T4 để chạy ba cấu hình β = 0,05 / 0,1 / 0,5. Tôi chọn T4 vì đây là GPU thực tế sẵn có trong Colab và bài lab yêu cầu trước hết một chuỗi SFT → DPO → đánh giá có thể kiểm chứng. Giữ một split cố định, reference là bản SFT đã merge và tiền tính log-prob của reference giúp giảm VRAM, đồng thời làm rõ rằng thay đổi cuối là do DPO chứ không phải do đổi dữ liệu đánh giá giữa chừng.

Kết quả xác nhận một phần quyết định này: 100 bước DPO chạy hết trong khoảng 29 phút, loss ghi lần đầu 0,6935 gần log 2 như dự đoán, và margin held-out cuối dương 0,0797 với accuracy 67%. Tuy nhiên, chi tiết làm tôi bất ngờ là reward của cả chosen và rejected đều tăng; hàm tự động gọi `INTENDED` nhưng hình dạng này chưa phải quỹ đạo lý tưởng chosen tăng/rejected giảm. Tập preference cũng thiên về câu chosen dài hơn 65,9%, vì vậy việc margin tăng riêng lẻ chưa đủ chứng minh câu trả lời tốt hơn. Nếu làm lại, tôi sẽ giữ nguyên split và seed, thêm phép chấm trên cặp đầu ra có độ dài gần nhau, rồi mới ưu tiên β-sweep hoặc RPO/ORPO. Cách đó kiểm tra giả thuyết "học sở thích" đối lập với "học viết dài" trước khi tiêu thêm GPU cho mô hình lớn hơn.

---

## 7. Bộ đo chuẩn (bonus NB6)

Chưa chạy IFEval, GSM8K hoặc Global-MMLU-vi; không có điểm hay sai số chuẩn để kết luận về alignment tax.

---

## 8. Biến thể loss (bonus NB3b)

Chỉ chạy DPO sigmoid với β = 0,1. Chưa chạy RPO, DPO-norm, LD-DPO hoặc ORPO nên không suy đoán biến thể nào thay đổi độ dài nhiều nhất từ thực nghiệm này.

---

## 9. GRPO (bonus NB7)

Chưa chạy GRPO; không có số liệu trước/sau để ước lượng hiệu quả hay nhiễu thống kê.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Reward gap held-out dương nhưng hai bản hòa ở 36/50 câu; tốt hơn trên log-prob của cặp preference chưa chắc đã tạo đầu ra khác biệt khi sinh greedy. Giám khảo cùng họ Qwen3 còn trượt sanity tiếng Việt, trong khi giám khảo Llama đạt 12/12.
