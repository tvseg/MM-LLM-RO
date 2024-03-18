import numpy as np
import seg_metrics.seg_metrics as sg

def calculate_score(label, pred,  score):
    if score == "dice":
        intersect = float(np.sum(pred.astype(int) * label.astype(int)))
        union = float(np.sum(pred.astype(int)) + np.sum(label.astype(int)))
        return (2 * intersect) / union

    elif score == "iou":
        intersect = float(np.sum(pred.astype(int) * label.astype(int)))
        union = float(np.sum(np.logical_or(pred, label).astype(int)))
        return (intersect / union)

    elif score == "acc":
        TP = float(np.sum(pred.astype(int) * label.astype(int)))
        TN = float(np.sum((~pred).astype(int) * (~label).astype(int)))
        if len(pred.shape) == 2:
            H, W = pred.shape
            return ((TP + TN) / (H * W))
        else:
            H, W, D = pred.shape
            return (TP + TN) / (H * W * D)

    elif score == "voe":
        intersect = float(np.sum(pred.astype(int) * label.astype(int)))
        union = float(np.sum(np.logical_or(pred, label).astype(int)))
        return 1 - (intersect / union)

    elif score == "sens":
        TP = float(np.sum(pred.astype(int) * label.astype(int)))
        FN = float(np.sum((~pred).astype(int) * (label).astype(int)))
        if TP + FN == 0:
            return 0
        return TP/(TP+FN)

    elif score == "ppv":
        TP = float(np.sum(pred.astype(int) * label.astype(int)))
        FP = float(np.sum(pred.astype(int) * (~label).astype(int)))
        if TP + FP == 0:
            return 0

        return TP / (TP + FP)

    elif score == "hd":
        hd = sg.write_metrics(labels=[1],  # exclude background if needed
                  gdth_img=label,
                  pred_img=pred,
                  metrics='hd')  

        return hd[0]['hd'][0] * 0.1
