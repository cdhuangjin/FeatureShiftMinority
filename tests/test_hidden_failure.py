"""Hidden minority failure threshold logic."""


def _hidden(auroc_drop, auprc_drop, rec_drop, auroc_max=0.03, auprc_max=0.05, rec_min=0.10):
    return {
        "hidden_auroc": auroc_drop <= auroc_max and rec_drop >= rec_min,
        "hidden_auprc": auprc_drop <= auprc_max and rec_drop >= rec_min,
    }


def test_hidden_auroc_detected():
    h = _hidden(auroc_drop=0.02, auprc_drop=0.10, rec_drop=0.20)
    assert h["hidden_auroc"] and not h["hidden_auprc"]


def test_hidden_auprc_detected():
    h = _hidden(auroc_drop=0.20, auprc_drop=0.04, rec_drop=0.30)
    assert h["hidden_auprc"] and not h["hidden_auroc"]


def test_no_hidden_when_recall_stable():
    h = _hidden(auroc_drop=0.01, auprc_drop=0.01, rec_drop=0.05)
    assert not h["hidden_auroc"] and not h["hidden_auprc"]

