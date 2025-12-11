# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
import torch
from models import LeanDeepHubert, HungarianTextDataset
from transformers import AutoTokenizer
import pandas
import config
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from utils import format_baseline_log, format_model_performance, setup_logger
from baseline import fit_eval_baseline

logger = setup_logger()

def predict_understandability(paragraph):
    inputs = tokenizer(
        paragraph,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )

    # Safety: drop token_type_ids if tokenizer still returns it
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        prediction = int(torch.argmax(logits, dim=-1).item())

    return prediction
       

if __name__ == "__main__":

    logger.info(" ****************************** INFERENCE DEMONSTRATION STARTED ******************************")


    model = LeanDeepHubert(config.MODEL_NAME, num_labels = config.NUM_LABELS)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model parameters loaded from {config.MODEL_SAVE_PATH} for inference.")
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH)
    logger.info(f"Tokenizer loaded from {config.TOKENIZER_SAVE_PATH}.")
    model.eval()

    input_paragraph = "Az Alza a reklamáció beérkezését követően haladéktalanul létrehozza a reklamáció sorszámát („RMA szám”). Az RMA szám létrehozása csak a reklamáció beérkezését erősíti meg, a reklamációs igény elbírálására ebben a fázisban nem kerül sor. Ezt követően az Alza „Reklamációs igénylap” elnevezésű dokumentumot készít a reklamációról. Ezen dokumentum felel meg a fogyasztó és vállalkozás közötti szerződés keretében eladott dolgokra vonatkozó szavatossági és jótállási igények intézésének eljárási szabályairól szóló 19/2014. (IV. 29.) NGM rendelet 4 §-a szerinti jegyzőkönyvnek. Ezen Reklamációs Igénylap tartalmazza a Fogyasztó adatait, a reklamáció tárgyát képző termék elnevezését, vételárát, a Felek között létrejött szerződésről kiállított számlán szereplő dátumot, a hiba bejelentésének időpontját, a hiba leírását és a Fogyasztó által érvényesíteni kívánt reklamációs igényt is."

    logger.info("Predicting understandability for the input paragraph without newline characters:")
    logger.info(
        (
            "\n Az Alza a reklamáció beérkezését követően haladéktalanul létrehozza a reklamáció "
            "sorszámát („RMA szám”). Az RMA szám létrehozása csak a reklamáció beérkezését erősíti "
            "meg, a reklamációs igény elbírálására ebben a fázisban nem kerül sor. Ezt követően az Alza "
            "„Reklamációs igénylap” elnevezésű dokumentumot készít a reklamációról. Ezen dokumentum "
            "felel meg a fogyasztó és vállalkozás közötti szerződés keretében eladott dolgokra vonatkozó "
            "szavatossági és jótállási igények intézésének eljárási szabályairól szóló 19/2014. (IV. 29.) NGM "
            "rendelet 4 §-a szerinti jegyzőkönyvnek. Ezen Reklamációs Igénylap tartalmazza a Fogyasztó "
            "adatait, a reklamáció tárgyát képző termék elnevezését, vételárát, a Felek között létrejött "
            "szerződésről kiállított számlán szereplő dátumot, a hiba bejelentésének időpontját, a hiba "
            "leírását és a Fogyasztó által érvényesíteni kívánt reklamációs igényt is."
        )
    )

    prediction = predict_understandability(input_paragraph)

    logger.info(f"Predicted understandability: {prediction}")