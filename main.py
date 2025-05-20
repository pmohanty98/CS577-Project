import json
import numpy as np
import torch
import pandas as pd
import re
import textstat
import argparse
import seaborn as sns
import os
from transformers import PegasusTokenizer
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from evaluate import load
rouge_metric = load("rouge")
# If you want coverage ratio, we'll just do a simple function
def coverage_ratio(gen_text, input_text):
    """
    A naive coverage ratio: the fraction of gen_text tokens that appear in input_text.
    This is a simplistic approach, might not be super meaningful but for demonstration.
    """
    gen_tokens = gen_text.lower().split()
    input_tokens = set(input_text.lower().split())
    matches = sum(1 for tok in gen_tokens if tok in input_tokens)
    if len(gen_tokens) == 0:
        return 0.0
    return matches / len(gen_tokens)



def generate_summary(model, tokenizer, text, max_input_length=512, max_output_length=150):
    """
    A generic function to run seq2seq summarization.
    For T5, we might prepend 'summarize:' but let's keep it optional.
    """
    # For T5, often you do "summarize: <text>"
    # For BART or Pegasus, you can just feed text directly.
    # We'll do T5 style here:

    if "t5" in model_name.lower():
        input_text = "summarize: " + text
    else:
        input_text = text


    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        model.to("cuda")

    summary_ids = model.generate(
        inputs,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def remove_placeholders(placeholder,text):
    # Replace three underscores placeholder with a space
    return text.replace(placeholder, "").replace("  ", " ").strip()


def remove_boxes(text):
    # Regex pattern to remove all placeholders enclosed within square brackets
    cleaned_text = re.sub(r'\[.*?\]', '', text)

    # Optionally, clean excessive whitespace resulting from the removals
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def parse_hospital_summaries(json_path):
    """
    Reads the 'hospital_summarization.json' file, returns a list of dicts
    with 'cleaned_instruct' and 'gold_summary'.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed_data = []
    for entry in data:
        instruct_text = entry.get("instruct", "")
        gold_summary = entry.get("answer", "")

        # Remove lines like 'age:', 'gender:', etc.
        placeholder="___"
        cleaned_instruct = remove_placeholders(placeholder,instruct_text)
        cleaned_summary = remove_placeholders(placeholder, gold_summary)

        placeholder = "\n"
        cleaned_instruct = remove_placeholders(placeholder,cleaned_instruct)
        cleaned_summary = remove_placeholders(placeholder, cleaned_summary)

        parsed_data.append({
            "instructions": cleaned_instruct,
            "gold_summary": cleaned_summary
        })

    return parsed_data

def parse_patient_education(csv_path):
    df = pd.read_csv(csv_path)
    data_pairs = []
    for _, row in df.iterrows():

        cleaned_instruct = remove_boxes(row['discharge_instruction'])
        cleaned_summary = remove_boxes( row['discharge_summary'])

        data_pairs.append({
            'instructions': cleaned_instruct,
            'gold_summary': cleaned_summary
        })
    return data_pairs

def average_characters_per_token(text):
    tokens = text.split()
    if not tokens:
        return 0.0
    total_chars = sum(len(token) for token in tokens)
    return total_chars / len(tokens)

def augmented_clinical_notes():
    df = pd.read_json("hf://datasets/AGBonnet/augmented-clinical-notes/augmented_notes_30K.jsonl", lines=True)
    data_pairs = []
    for _, row in df.iterrows():

        cleaned_instruct = row['full_note']
        cleaned_summary =  row['summary']

        data_pairs.append({
            'instructions': cleaned_instruct,
            'gold_summary': cleaned_summary
        })
    return data_pairs




# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hospital') #50,000
    parser.add_argument('--model', type=str, default='t5-small')   #1
    parser.add_argument('--GPU', type=int, default=1) #50,000
    args = parser.parse_args()
    keyword=args.dataset

    if keyword=='hospital':
        dataset_path = "Hospitalization-Summarization.json"
        parsed_entries = parse_hospital_summaries(dataset_path)
    elif keyword=='patient':
        dataset_path = "Patient-Education.csv"
        parsed_entries = parse_patient_education(dataset_path)
    else:
        parsed_entries = augmented_clinical_notes()

    print(f"Loaded {len(parsed_entries)} entries.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # 3) Choose a model (e.g. T5)



    model_name=args.model
    #model_name = "t5-small"  #  "facebook/bart-large-cnn", "google/pegasus-xsum"
    if "pegasus" in model_name:
        tokenizer= PegasusTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # If you want to do it on CPU or GPU:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 4) We'll store results
    all_rouge_scores = []
    all_cov_ratios = []
    all_bert_scores = []
    all_flesch_reading_ease = []
    all_flesch_kincaid_grade = []
    all_smog_index = []
    all_coverage_ratios = []


    # Let's do a small subset for demonstration
    # If you want to do all, remove [:10]
    for i, entry in enumerate(parsed_entries):
        input_text = entry["instructions"]
        gold_text = entry["gold_summary"]



        # Generate summary
        gen_text = generate_summary(model, tokenizer, input_text)

        bert_score_metric = load("bertscore")


        # Compute BERTScore between generated and gold
        bert_score = bert_score_metric.compute(
            predictions=[gen_text],
            references=[gold_text],
            lang="en"  # set correct language
        )
        bert_score = bert_score['f1'][0]
        #print(f"BERTScore F1: {bert_score['f1'][0]:.3f}")




        # Assume `generated_summary` is your model output
        reading_ease_score = textstat.flesch_reading_ease(gen_text)
        grade_level = textstat.flesch_kincaid_grade(gen_text)

        #print(f"Flesch Reading Ease Score: {reading_ease_score:.3f}")
        #print(f"Flesch-Kincaid Grade Level: {grade_level:.3f}")

        #smog_index = textstat.smog_index(gen_text)
        smog_index = average_characters_per_token(gen_text)


        #print(f"SMOG Grade Level: {smog_index:.3f}")


        # Evaluate ROUGE
        # The huggingface load_metric("rouge") expects lists
        rouge_scores = rouge_metric.compute(
            predictions=[gen_text],
            references=[gold_text]
        )

        # We might only focus on ROUGE-L or something, let's pick them all
        # Typically you might store them in a dict
        # e.g. rouge_scores["rouge1"].mid.fmeasure
        r1 = rouge_scores["rouge1"]  # instead of .mid.fmeasure
        r2 = rouge_scores["rouge2"]
        rl = rouge_scores["rougeL"]

        # coverage ratio
        cov = coverage_ratio(gen_text, input_text)

        # store


        all_rouge_scores.append((r1, r2, rl))
        all_cov_ratios.append(cov)

        all_bert_scores.append(bert_score)
        all_flesch_reading_ease.append(reading_ease_score)
        all_flesch_kincaid_grade.append(grade_level)
        all_smog_index.append(smog_index)


        # Print a snippet
        print(f"=== Example {i} ===")
        print(f"Input (truncated): {input_text[:200]}...")
        print(f"Gold Summary: {gold_text[:150]}...")
        print(f"Generated Summary: {gen_text}\n")
        print(f"ROUGE-1: {r1:.3f}, ROUGE-2: {r2:.3f}, ROUGE-L: {rl:.3f}, Coverage: {cov:.3f}")
        print(f"Bert-Score: {bert_score:.3f}, Flesch Reading Ease Score: {reading_ease_score:.3f}, Flesch-Kincaid Grade Level: {grade_level:.3f}, CPT: {smog_index:.3f}")
        print("==============\n")

    # 5) Summarize overall scores
    avg_r1 = np.mean([x[0] for x in all_rouge_scores])
    std_r1=np.std([x[0] for x in all_rouge_scores])

    avg_r2 = np.mean([x[1] for x in all_rouge_scores])
    std_r2 = np.std([x[1] for x in all_rouge_scores])

    avg_rl = np.mean([x[2] for x in all_rouge_scores])
    std_rl = np.std([x[2] for x in all_rouge_scores])
    avg_cov = np.mean(all_cov_ratios)
    std_cov = np.std(all_cov_ratios)

    print(f"Average ROUGE-1: {avg_r1:.3f} +_ "+ str(std_r1))
    print(f"Average ROUGE-2: {avg_r2:.3f} +_ "+str(std_r2))
    print(f"Average ROUGE-L: {avg_rl:.3f} +_ "+str(std_rl))
    print(f"Average Coverage Ratio: {avg_cov:.3f} +_ "+str(std_cov))


    print(f"Average Bert-Score: {np.mean(all_bert_scores):.3f} +_ "+ str(np.std(all_bert_scores)))
    print(f"Average FRES: {np.mean(all_flesch_reading_ease):.3f} +_ "+str(np.std(all_flesch_reading_ease)))
    print(f"Average FKGL: {np.mean(all_flesch_kincaid_grade):.3f} +_ "+str(np.std(all_flesch_kincaid_grade)))
    print(f"Average CPT: {np.mean(all_smog_index):.3f} +_ "+str(np.std(all_smog_index)))

    # Save each metric list to a .npy file
    np.save("rouge1_list.npy", [x[0] for x in all_rouge_scores])
    np.save("rouge2_list.npy", [x[1] for x in all_rouge_scores])
    np.save("rougeL_list.npy", [x[2] for x in all_rouge_scores])
    np.save("coverage_list.npy", all_cov_ratios)
    np.save("bertscore_list.npy", all_bert_scores)
    np.save("fres_list.npy", all_flesch_reading_ease)
    np.save("fkgl_list.npy", all_flesch_kincaid_grade)
    np.save("cpt_list.npy", all_smog_index)




    df = pd.DataFrame({
        "ROUGE-L": [x[2] for x in all_rouge_scores],  # Assuming x = (ROUGE-1, ROUGE-2, ROUGE-L)
        "Coverage": all_cov_ratios,
        "BERTScore": all_bert_scores,
        "FRES": all_flesch_reading_ease,
        "FKGL": all_flesch_kincaid_grade,
        "CPT": all_smog_index,
    })

    sns.set(style="white", font_scale=1.1)
    pairplot = sns.pairplot(df, corner=True, kind="scatter", diag_kind="hist", plot_kws={"alpha": 0.5})
    pairplot.fig.suptitle("Pairwise Distributions for Hospitalization Dataset Metrics", y=1.02)

    plt.tight_layout()
    plt.show()

