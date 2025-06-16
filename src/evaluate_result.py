from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, roc_auc_score


def evaluate(veri_results, label_n=2):

    entailment_score_lst = []
    for claim_res in tqdm(veri_results):
        claim_res['binary_label'] = 1 if claim_res['annot_label'] == 'supported' else 0
        claim_preds = [v for k,v in claim_res['claim_verification_result'].items()]
        if label_n == 2:
            label2score = {
                "supported": 1.0,
                "unsupported": 0.0,
            }
        elif label_n == 3:
            label2score = {
                "supported": 1.0,
                "unsupported": 0.0,
                "partially supported": 0.5,
            }
        claim_preds = [label2score[v] for v in claim_preds]
        entailment_score_lst.append(sum(claim_preds) / len(claim_preds))
    
    label_lst = [item['binary_label'] for item in veri_results]
    
    # Compute the F1 score, precision, recall, accuracy, AUROC
    sep_claim_lst = [len(item['claim_verification_result']) for item in veri_results]
    avg_sep_claim = sum(sep_claim_lst) / len(sep_claim_lst)

    # Convert entailment scores to binary labels
    binary_preds = [1 if score >= 0.5 else 0 for score in entailment_score_lst]

    # Compute F1 score, precision, recall, accuracy, AUROC
    f1 = f1_score(label_lst, binary_preds)
    precision = precision_score(label_lst, binary_preds)
    recall = recall_score(label_lst, binary_preds)
    accuracy = accuracy_score(label_lst, binary_preds)
    balanced_accuracy = balanced_accuracy_score(label_lst, binary_preds)
    auroc = roc_auc_score(label_lst, entailment_score_lst)

    # All result round to 4 decimal places
    f1 = round(f1, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    accuracy = round(accuracy, 4)
    balanced_accuracy = round(balanced_accuracy, 4)
    auroc = round(auroc, 4)
    avg_sep_claim = round(avg_sep_claim, 4)

    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUROC: {auroc}")
    print(f"Claim Count: {avg_sep_claim}")

    metrics_dict = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "claim_count": avg_sep_claim
    }

    return metrics_dict

