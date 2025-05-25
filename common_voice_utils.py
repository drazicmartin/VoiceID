from datasets import load_dataset, Audio
import random
import torch
import torchaudio
import speechbrain as sb
import sys
import numpy as np
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from functools import partial
from speechbrain.dataio.dataset import DynamicItemDataset

LANGUAGES = {'ab': 'Abkhaz', 'ace': 'Acehnese', 'ady': 'Adyghe', 'af': 'Afrikaans', 'am': 'Amharic', 'an': 'Aragonese', 'ar': 'Arabic', 'arn': 'Mapudungun', 'as': 'Assamese', 'ast': 'Asturian', 'az': 'Azerbaijani', 'ba': 'Bashkir', 'bas': 'Basaa', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Tibetan', 'br': 'Breton', 'bs': 'Bosnian', 'bxr': 'Buryat', 'byv': 'Medumba', 'ca': 'Catalan', 'cak': 'Kaqchikel', 'ckb': 'Central Kurdish', 'cnh': 'Hakha Chin', 'co': 'Corsican', 'crh': 'Crimean Tatar', 'cs': 'Czech', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'dag': 'Dagbani', 'de': 'German', 'dsb': 'Sorbian, Lower', 'dv': 'Dhivehi', 'dyu': 'Dioula', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'ewo': 'Ewondo', 'fa': 'Persian', 'ff': 'Fulah', 'fi': 'Finnish', 'fo': 'Faroese', 'fr': 'French', 'fuf': 'Pular Guinea', 'fy-NL': 'Frisian', 'ga-IE': 'Irish', 'gl': 'Galician', 'gn': 'Guarani', 'gom': 'Goan Konkani', 'gu-IN': 'Gujarati', 'guc': 'Wayuunaiki', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'hil': 'Hiligaynon', 'hr': 'Croatian', 'hsb': 'Sorbian, Upper', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy-AM': 'Armenian', 'hyw': 'Armenian Western', 'ia': 'Interlingua', 'id': 'Indonesian', 'ie': 'Interlingue', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'izh': 'Izhorian', 'ja': 'Japanese', 'jbo': 'Lojban', 'jv': 'Javanese', 'ka': 'Georgian', 'kaa': 'Karakalpak', 'kab': 'Kabyle', 'kbd': 'Kabardian', 'ki': 'Kikuyu', 'kk': 'Kazakh', 'km': 'Khmer', 'kmr': 'Kurmanji Kurdish', 'kn': 'Kannada', 'knn': 'Konkani (Devanagari)', 'ko': 'Korean', 'kpv': 'Komi-Zyrian', 'kw': 'Cornish', 'ky': 'Kyrgyz', 'lb': 'Luxembourgish', 'lg': 'Luganda', 'lij': 'Ligurian', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'ltg': 'Latgalian', 'lv': 'Latvian', 'lzz': 'Laz', 'mai': 'Maithili', 'mdf': 'Moksha', 'mg': 'Malagasy', 'mhr': 'Meadow Mari', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mni': 'Meetei Lon', 'mos': 'Mossi', 'mr': 'Marathi', 'mrj': 'Hill Mari', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'myv': 'Erzya', 'nan-tw': 'Taiwanese (Minnan)', 'nb-NO': 'Norwegian Bokmål', 'nd': 'IsiNdebele (North)', 'ne-NP': 'Nepali', 'nhe': 'Eastern Huasteca Nahuatl', 'nhi': 'Western Sierra Puebla Nahuatl', 'nia': 'Nias', 'nl': 'Dutch', 'nn-NO': 'Norwegian Nynorsk', 'nr': 'IsiNdebele (South)', 'nso': 'Northern Sotho', 'ny': 'Chinyanja', 'nyn': 'Runyankole', 'oc': 'Occitan', 'om': 'Afaan Oromo', 'or': 'Odia', 'os': 'Ossetian', 'pa-IN': 'Punjabi', 'pap-AW': 'Papiamento (Aruba)', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'quc': "K'iche'", 'quy': 'Quechua Chanka', 'qvi': 'Kichwa', 'rm-sursilv': 'Romansh Sursilvan', 'rm-vallader': 'Romansh Vallader', 'ro': 'Romanian', 'ru': 'Russian', 'rw': 'Kinyarwanda', 'sah': 'Sakha', 'sat': 'Santali (Ol Chiki)', 'sc': 'Sardinian', 'scn': 'Sicilian', 'sco': 'Scots', 'sd': 'Sindhi', 'sdh': 'Southern Kurdish', 'shi': 'Shilha', 'si': 'Sinhala', 'sk': 'Slovak', 'skr': 'Saraiki', 'sl': 'Slovenian', 'snk': 'Soninke', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'ss': 'Siswati', 'st': 'Southern Sotho', 'sv-SE': 'Swedish', 'sw': 'Swahili', 'syr': 'Syriac', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tig': 'Tigre', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tn': 'Setswana', 'tok': 'Toki Pona', 'tr': 'Turkish', 'ts': 'Xitsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian', 'tyv': 'Tuvan', 'uby': 'Ubykh', 'udm': 'Udmurt', 'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Tshivenda', 'vec': 'Venetian', 'vi': 'Vietnamese', 'vmw': 'Emakhuwa', 'vot': 'Votic', 'wep': 'Westphalian', 'wo': 'Wolof', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'yue': 'Cantonese', 'zgh': 'Tamazight', 'zh-CN': 'Chinese (China)', 'zh-HK': 'Chinese (Hong Kong)', 'zh-TW': 'Chinese (Taiwan)', 'zu': 'Zulu', 'zza': 'Zaza'}

def process_sample(sample, hparams):
    target_len = int(hparams["sample_rate"] * hparams["sentence_len"])
    # Ensure it's a numpy array
    sig = sample['audio'][0]['array']
    if sig.shape[0] > target_len:
        start = random.randint(0, sig.shape[0] - target_len)
        sig = sig[start : start + target_len]
    elif sig.shape[0] < target_len:
        pad = target_len - sig.shape[0]
        sig = np.pad(sig, (0, pad))

    # sig = torch.tensor(sig, dtype=torch.float32)
    # sig = sig / torch.max(torch.abs(sig))
    # example {'id': 'id01754--WnrHrsxu6gk--00360_0.0_3.0', 'sig': tensor([ 0.0721,  0.0854,  0.0964,  ..., -0.1073, -0.0811, -0.0555]), 'spk_id_encoded': tensor([0])}

    sig = torch.tensor(sig, dtype=torch.float32)
    sig = sig / torch.max(torch.abs(sig))  # Normalize the signal
    spk_id_encoded = torch.tensor(sample['spk_id_encoded'], dtype=torch.int64)
    return {
        "sig": [sig],
        "spk_id_encoded": spk_id_encoded,
        "id": [Path(sample["path"][0]).stem],
    }

def get_single_dataset(hparams, lang="fr", split="train", debug=False):
    assert lang in LANGUAGES

    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # cache_dir="J:/.hf_cache"
    cv_17 = load_dataset("mozilla-foundation/common_voice_17_0", lang, split=split)
    if debug:
        cv_17 = cv_17.take(20)
    cv_17 = cv_17.cast_column("audio", Audio(sampling_rate=hparams["sample_rate"]))
    cv_17 = cv_17.rename_column("client_id", "spk_id")

    unique_spk_ids = cv_17.unique('spk_id')
    lab_enc_file = Path(hparams["save_folder"]) / f"{split}_label_encoder.txt"
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_iterables=[unique_spk_ids],
    )

    label_encoder.ignore_len()

    print(f"Number of speakers in {split} set: {len(unique_spk_ids)}")

    cv_17 = cv_17.map(lambda sample: {"spk_id_encoded": label_encoder.encode_sequence_torch([sample['spk_id']])})

    cv_17.set_transform(
        partial(
            process_sample,
            hparams=hparams,
        )
    )
    return cv_17, label_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    get_single_dataset(hparams)