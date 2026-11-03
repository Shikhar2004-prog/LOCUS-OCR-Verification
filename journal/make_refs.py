"""Write j_references.tex: the 25 references in APA 7 style, numbered in order of first citation.

Order is read from locus_journal.aux (run pdflatex once first); unknown order falls back to the dict order.
"""
import os
import re

A = r"\textit"
REFS = {
    "fu2025ocrbenchv2": r"Fu, L., Kuang, Z., Song, J., Huang, M., Yang, B., Li, Y., Zhu, L., Luo, Q., Wang, X., Lu, H., Li, Z., Tang, G., Shan, B., Lin, C., Liu, Q., Wu, B., Feng, H., Liu, H., Huang, C., \ldots\ Bai, X. (2025). OCRBench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. In \textit{Advances in Neural Information Processing Systems, Datasets and Benchmarks Track}.",
    "he2025seeing": r"He, Z., Zhang, C., Wu, Z., Chen, Z., Zhan, Y., Li, Y., Zhang, Z., Wang, X., \& Qiu, M. (2025). Seeing is believing? Mitigating OCR hallucinations in multimodal large language models. In \textit{Advances in Neural Information Processing Systems}.",
    "gardella2026lowcer": r"Gardella, M., Mari\~no, C., Belzarena, D., Ram\'irez, I., Randall, G., \& Morel, J.-M. (2026). \textit{When low CER is not enough: An analysis of hallucinations in vision-language OCR systems on historical Uruguayan documents} (arXiv:2607.24077). arXiv.",
    "bautista2022parseq": r"Bautista, D., \& Atienza, R. (2022). Scene text recognition with permuted autoregressive sequence models. In \textit{Proceedings of the European Conference on Computer Vision (ECCV)} (pp.~178--196). Springer.",
    "li2023trocr": r"Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., \& Wei, F. (2023). TrOCR: Transformer-based optical character recognition with pre-trained models. In \textit{Proceedings of the AAAI Conference on Artificial Intelligence}, \textit{37}(11), 13094--13102.",
    "xiao2024florence": r"Xiao, B., Wu, H., Xu, W., Dai, X., Hu, H., Lu, Y., Zeng, M., Liu, C., \& Yuan, L. (2024). Florence-2: Advancing a unified representation for a variety of vision tasks. In \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)} (pp.~4818--4829).",
    "bai2025qwen25vl": r"Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. (2025). \textit{Qwen2.5-VL technical report} (arXiv:2502.13923). arXiv.",
    "zhang2025docrinspector": r"Zhang, Q., Zhang, J., Ren, Z., Ouyang, L., Wen, Z., Niu, J., Qu, Y., Wang, B., Chow, K.-H., He, C., \& Zhang, W. (2025). \textit{DOCR-Inspector: Fine-grained and automated evaluation of document parsing with VLM} (arXiv:2512.10619). arXiv.",
    "dubey2025hindi": r"Dubey, S., Mittal, K., Behera, S. K., Ravikiran, M., Kumar, N., Shigwan, S., \& Saluja, R. (2025). Multi-feature graph convolution network for Hindi OCR verification. In \textit{Proceedings of the 1st Workshop on Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in Indian Languages (BHASHA)} (pp.~1--10). Association for Computational Linguistics.",
    "corbiere2019confidnet": r"Corbi\`ere, C., Thome, N., Bar-Hen, A., Cord, M., \& P\'erez, P. (2019). Addressing failure prediction by learning model confidence. In \textit{Advances in Neural Information Processing Systems} (Vol.~32).",
    "greif2025multimodal": r"Greif, G., Griesshaber, N., \& Greif, R. (2025). \textit{Multimodal LLMs for OCR, OCR post-correction, and named entity recognition in historical documents} (arXiv:2504.00414). arXiv.",
    "kanerva2025nofree": r"Kanerva, J., Ledins, C., K\"apyaho, S., \& Ginter, F. (2025). OCR error post-correction with LLMs in historical documents: No free lunches. In \textit{Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL)}.",
    "ehrmann2026hipe": r"Ehrmann, M., Boros, E., Opitz, J., Michail, A., Wagner, F., \& Clematide, S. (2026). \textit{ICDAR 2026 HIPE-OCRepair competition on LLM-assisted OCR post-correction for historical documents} (arXiv:2607.08143). arXiv.",
    "ocredr2026": r"Zhao, L., Liu, K., Yu, H., Zhan, J., Sun, C., \& Li, C. (2026). \textit{OCR-EDR: Rendering-aware diagnosis and repair for closed-loop OCR improvement} (arXiv:2609.03445). arXiv.",
    "wei2024got": r"Wei, H., Liu, C., Chen, J., Wang, J., Kong, L., Xu, Y., Ge, Z., Zhao, L., Sun, J., Peng, Y., Han, C., \& Zhang, X. (2024). \textit{General OCR theory: Towards OCR-2.0 via a unified end-to-end model} (arXiv:2409.01704). arXiv.",
    "poznanski2025olmocr": r"Poznanski, J., Rangapur, A., Borchardt, J., Dunkelberger, J., Huff, R., Lin, D., Wilhelm, C., Lo, K., \& Soldaini, L. (2025). \textit{olmOCR: Unlocking trillions of tokens in PDFs with vision language models} (arXiv:2502.18443). arXiv.",
    "saxena2026reading": r"Saxena, S., Tripathi, V., \& Abbas, S. (2026). Reading between the characters: Label-efficient OCR verification with graph neural networks. In \textit{Proceedings of NICCAI 2026}, Hyderabad, India (to appear).",
    "leng2024vcd": r"Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C., \& Bing, L. (2024). Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)} (pp.~13872--13882).",
    "brody2022gatv2": r"Brody, S., Alon, U., \& Yahav, E. (2022). How attentive are graph attention networks? In \textit{International Conference on Learning Representations (ICLR)}.",
    "jaume2019funsd": r"Jaume, G., Ekenel, H. K., \& Thiran, J.-P. (2019). FUNSD: A dataset for form understanding in noisy scanned documents. In \textit{International Conference on Document Analysis and Recognition Workshops (ICDARW)} (Vol.~2, pp.~1--6). IEEE.",
    "park2019cord": r"Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., \& Lee, H. (2019). CORD: A consolidated receipt dataset for post-OCR parsing. In \textit{Workshop on Document Intelligence at NeurIPS 2019}.",
    "huang2019sroie": r"Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., \& Jawahar, C. V. (2019). ICDAR2019 competition on scanned receipt OCR and information extraction. In \textit{International Conference on Document Analysis and Recognition (ICDAR)} (pp.~1516--1520). IEEE.",
    "marti2002iam": r"Marti, U.-V., \& Bunke, H. (2002). The IAM-database: An English sentence database for offline handwriting recognition. \textit{International Journal on Document Analysis and Recognition}, \textit{5}(1), 39--46.",
    "mishra2012iiit5k": r"Mishra, A., Alahari, K., \& Jawahar, C. V. (2012). Scene text recognition using higher order language priors. In \textit{Proceedings of the British Machine Vision Conference (BMVC)}.",
    "smith2007tesseract": r"Smith, R. (2007). An overview of the Tesseract OCR engine. In \textit{Ninth International Conference on Document Analysis and Recognition (ICDAR)} (Vol.~2, pp.~629--633). IEEE.",
}

here = os.path.dirname(os.path.abspath(__file__))
order = []
aux = os.path.join(here, "locus_journal.aux")
if os.path.exists(aux):
    for m in re.finditer(r"\\citation\{([^}]*)\}", open(aux).read()):
        for k in m.group(1).split(","):
            k = k.strip()
            if k in REFS and k not in order:
                order.append(k)
missing = [k for k in REFS if k not in order]
order += missing
unknown = set(re.findall(r"\\citation\{([^}]*)\}", open(aux).read()) if os.path.exists(aux) else [])
unknown = {k.strip() for g in unknown for k in g.split(",")} - set(REFS)
if unknown:
    print("CITED BUT NOT IN LIST:", unknown)
if missing and os.path.exists(aux):
    print("IN LIST BUT NEVER CITED:", missing)
out = [r"\begingroup\footnotesize\setlength{\parskip}{0pt}", r"\renewcommand{\refname}{\fontsize{12}{12}\selectfont\textbf{References}}",
       r"\begin{thebibliography}{99}\setlength{\itemsep}{1pt}"]
out += [rf"\bibitem{{{k}}} {REFS[k]}" for k in order]
out += [r"\end{thebibliography}", r"\endgroup"]
open(os.path.join(here, "j_references.tex"), "w").write("\n".join(out) + "\n")
print(len(order), "references written")
