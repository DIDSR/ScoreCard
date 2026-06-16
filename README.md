<p align="center">
 <h1 align="center">GenAI ScoreCard: A Data Quality Evaluation Toolbox for Synthetic Medical Data</h1>
</p>

<p align="center">
 <img src="imgs/ScoreCard.png">
</p>

## Overview

**`GenAI ScoreCard`** is an open-source Python toolbox that implements the **7 Cs evaluation framework** for assessing the quality of Synthetic Medical Data.

Synthetic medical data holds significant promise for advancing AI development in healthcare — particularly for addressing data scarcity, privacy constraints, and underrepresentation of rare diseases and populations. However, the quality of synthetic data is critical: poor-quality synthetic data can introduce biases, violate clinical constraints, and ultimately compromise the safety and effectiveness of AI models trained or tested on such data.

This toolbox provides a **quantitative and reproducible** approach to evaluating Synthetic Medical Data across seven clinically relevant criteria (the **7 Cs**), and supports the generation of a structured **Scorecard** to accompany synthetic datasets.

For more information, please contact: **[Seyed.Kahaki@fda.hhs.gov](mailto:Seyed.Kahaki@fda.hhs.gov)**

---

## The 7 Cs Framework

The SMD ScoreCard evaluates synthetic medical data across the following seven dimensions:

1. **Congruence** — Measures the degree to which the distribution of synthetic data aligns with the distribution of real patient data (e.g., using Fréchet Inception Distance, Cosine Similarity, Jensen-Shannon Divergence).

2. **Coverage** — Evaluates the extent to which SMD captures the variability, range, and novelty inherent in patient data (e.g., using Convex Hull Volume, Recall, Vendi Score, Entropy).

3. **Constraint** — Assesses adherence to known anatomical, biological, clinical, geometric, or user-defined constraints (e.g., using Constraint Violation Rate, Distance to Constraint Boundary).

4. **Completeness** — Evaluates whether the generated data contains all necessary details relevant to the intended task (e.g., using Proportion of Required Fields, Missing Data Percentage).

5. **Compliance** — Reports adherence to privacy standards, format guidelines (e.g., DICOM), and relevant local and international requirements (e.g., using Differential Privacy Score, K-Anonymity Level, Re-identification Risk metrics).

6. **Comprehension** — Evaluates the transparency and clarity of the data generation process and accompanying documentation (e.g., using Documentation Clarity Score).

7. **Consistency** — Assesses the stability of SMD quality metrics across demographic subgroups, disease classes, or over time (e.g., using Variance, ANOVA, Maximum-Minimum Difference).

---

## Synthetic Medical Data Scorecard Structure

In addition to quantitative evaluation, this toolbox supports generation of a structured **SMD Scorecard** report containing the following sections:

1. **General Information** — Dataset name, modality, size, labels, licensing, and point of contact.
2. **Data Quality Evaluation (7 Cs)** — Quantitative results for each of the seven criteria.
3. **Task-based Evaluation** — Performance metrics for specific downstream tasks.
4. **Human-based Evaluation** — Results from qualitative reader studies and failure case analysis.
5. **Ethical Considerations, Limitations & Recommendations** — Known biases, limitations, and best practices.
6. **Dataset Usage** — Repository links, preprocessing requirements, and user documentation.
7. **Training & Validation Process** — Description of the synthetic data generation pipeline.
8. **Reference Dataset Information** — Key details of the patient dataset used for comparison.

---

## Getting Started

### Installation

Clone this repository and navigate to the project directory:

---

## Contact

For any inquiries, suggestions, or collaborative opportunities, please contact Seyed Kahaki, Elim Thompson, or Aldo Badano either via this GitHub repo or via email (seyed.kahaki@fda.hhs.gov, yeelamelim.thompson@fda.hhs.gov or aldo.badano@fda.hhs.gov).

---

