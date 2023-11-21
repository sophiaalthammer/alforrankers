
from matchmaker.losses.lambdarank import *
from matchmaker.losses.listnet import *
from matchmaker.losses.ranknet import *
from matchmaker.losses.msmargin import *
from matchmaker.losses.teacher_kldiv_list import *
from matchmaker.losses.teacher_kldiv_pointwise import *
from matchmaker.losses.teacher_mse_pointwise import *
from matchmaker.losses.teacher_ranknetweighted import *
from matchmaker.losses.teacher_hinge_weighted import *
from matchmaker.losses.teacher_mse_ranknet import *
from matchmaker.losses.QA_StartEndCrossEntropy import *
from matchmaker.losses.mlmloss import *
from matchmaker.losses.diversify_pos import *
from matchmaker.losses.max_length import *
from matchmaker.losses.qg_CE_logits import *
from matchmaker.losses.qg_kl import *
from matchmaker.losses.qg_KD_margin import *
from matchmaker.losses.qg_claps import *

def merge_loss(losses, log_vars):
    loss = torch.zeros(1,device=log_vars.device)
    weighted_losses = []
    for l in range(len(losses)):
        precision = torch.exp(-log_vars[l])
        wl = torch.sum(precision * losses[l] + log_vars[l], -1)
        loss += wl
        weighted_losses.append(wl.detach())
    return torch.mean(loss),weighted_losses

def get_loss(config):

    use_list_loss=False
    use_inbatch_list_loss=False
    qa_loss=None
    inbatch_loss=None
    inbatch_loss_query = None
    inbatch_loss_query_length = None

    if config["loss"] == "margin-mse":
        loss = MSMarginLoss()
    elif config["loss"] == "MSETeacherPointwise":
        loss = MSETeacherPointwise()
    elif config["loss"] == "MSETeacherPointwisePassages":
        loss = MSETeacherPointwisePassages()
    elif config["loss"] == "MarginMSE_InterPassageLoss":
        loss = MarginMSE_InterPassageLoss()
    elif config["loss"] == "KLDivTeacherPointwise":
        loss = KLDivTeacherPointwise()
    elif config["loss"] == "RankNetTeacher":
        loss = RankNetTeacher()
    elif config["loss"] == "MSERanknetTeacher":
        loss = MSERanknetTeacher()
    elif config["loss"] == "MLMLoss":
        loss = MLMLoss()

    elif config["loss"] == "ranknet":
        loss = RankNetLoss()
    elif config["loss"] == "margin":
        loss = torch.nn.MarginRankingLoss(margin=1, reduction='mean')
    elif config["loss"] == "weighted-hinge":
        loss = WeightedHingeLoss()
    elif config["loss"] == "mrr":
        loss = SmoothMRRLoss()
        use_list_loss = True
    elif config["loss"] == "listnet":
        loss = ListNetLoss()
        use_list_loss = True
    elif config["loss"] == "lambdarank":
        loss = LambdaLoss("ndcgLoss2_scheme")
        use_list_loss = True
    elif config["loss"] == "qg-cross-entropy":
        loss = CrossEntropyPointLoss(config)
    elif config["loss"] == "qg-cross-entropy-pn":
        loss = CrossEntropyPNLoss(config)
    elif config["loss"] == "qg-cross-entropy-kd":
        loss = CrossEntropyScoresLoss(config)
    elif config["loss"] == "qg-cross-entropy-kd-pn":
        loss = CrossEntropyScoresPNLoss(config)
    elif config["loss"] == "qg-cross-entropy-kl":
        loss = KLDiv()
    elif config["loss"] == "claps":
        loss = Claps()
    else:
        raise Exception("Loss not known")

    if config["train_qa_spans"]:
        if config["qa_loss"] == "StartEndCrossEntropy":
            qa_loss = QA_StartEndCrossEntropy()
        else:
            raise Exception("QA-Loss not known, qa_loss must be set with train_qa_spans")


    if config["in_batch_negatives"]:
        if config["in_batch_neg_loss"] == "ranknet":
            inbatch_loss = RankNetLoss()
        elif config["in_batch_neg_loss"] == "margin":
            inbatch_loss = torch.nn.MarginRankingLoss(margin=1, reduction='mean')
        elif config["in_batch_neg_loss"] == "margin-mse":
            inbatch_loss = MSMarginLoss()
        elif config["in_batch_neg_loss"] == "KLDivTeacherList":
            inbatch_loss = KLDivTeacherList()
            use_inbatch_list_loss = True
        elif config["in_batch_neg_loss"] == "listnet":
            inbatch_loss = ListNetLoss()
            use_inbatch_list_loss = True
        elif config["in_batch_neg_loss"] == "lambdarank":
            inbatch_loss = LambdaLossTeacher("ndcgLoss2_scheme")
            use_inbatch_list_loss = True
        elif config["in_batch_neg_loss"] == "qg-cross-entropy":
            inbatch_loss = CrossEntropyPointLoss(config)
        elif config["in_batch_neg_loss"] == "qg-cross-entropy-kd":
            inbatch_loss = CrossEntropyScoresLoss(config)
        elif config["in_batch_neg_loss"] == "qg-kl":
            inbatch_loss = KLDiv()
        else:
            raise Exception("In-batch-Loss not known, in_batch_neg_loss must be set with in_batch_negatives")

    if config["in_batch_queries"]:
        if config["in_batch_query_loss"] == "divquery":
            inbatch_loss_query = DivQueryLoss()
        else:
            raise Exception("In-batch-Query-Loss not known, in_batch_query_loss must be set with in_batch_queries")

    if config["intra_batch_queries"]:
        if config["intra_batch_query_loss"] == "divquery":
            inbatch_loss_query = DivQueryLoss()
        else:
            raise Exception("Intra-batch-Query-Loss not known, intra_batch_query_loss must be set with intra_batch_query_loss")

    if config["in_batch_queries_length"]:
        if config["in_batch_query_length_loss"] == "maxlength":
            inbatch_loss_query_length = MaxLengthLoss()
        else:
            raise Exception("In-batch-Query-Length-Loss not known, in_batch_query_length_loss must be set with in_batch_query_length_loss")

    return loss, qa_loss, inbatch_loss, use_list_loss,use_inbatch_list_loss, inbatch_loss_query, inbatch_loss_query_length