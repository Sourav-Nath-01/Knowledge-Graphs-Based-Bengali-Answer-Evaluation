"""
dataset.py
==========
Dataset loading and adversarial augmentation for the Bengali Answer Evaluation System.
"""

import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split


# ─── Adversarial examples ─────────────────────────────────────────────────────
_ADVERSARIAL_SAMPLES = [
    # --- Karak Reversal ---
    {'question': 'কে রাবণকে বধ করেছিলেন?', 'reference_answer': 'রাম রাবণকে বধ করেছিলেন।', 'student_answer': 'রাবণ রামকে বধ করেছিলেন।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'কে বাঘ শিকার করেছিল?', 'reference_answer': 'শিকারী বাঘ শিকার করেছিল।', 'student_answer': 'বাঘ শিকারীকে শিকার করেছিল।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে চিঠি লিখেছিল?', 'reference_answer': 'রহিম চিঠি লিখেছিল।', 'student_answer': 'চিঠি রহিমকে লিখেছিল।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'পুলিশ কাকে ধরেছে?', 'reference_answer': 'পুলিশ চোরকে ধরেছে।', 'student_answer': 'চোর পুলিশকে ধরেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে ভাত খাচ্ছে?', 'reference_answer': 'ছেলেটি ভাত খাচ্ছে।', 'student_answer': 'ভাত ছেলেটিকে খাচ্ছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে কাকে সাহায্য করেছে?', 'reference_answer': 'ডাক্তার রোগীকে সাহায্য করেছে।', 'student_answer': 'রোগী ডাক্তারকে সাহায্য করেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে ইঁদুর ধরেছে?', 'reference_answer': 'বিড়াল ইঁদুর ধরেছে।', 'student_answer': 'ইঁদুর বিড়ালকে ধরেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'কে সাপ মেরেছে?', 'reference_answer': 'কৃষক সাপ মেরেছে।', 'student_answer': 'সাপ কৃষককে মেরেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে বই পড়ছে?', 'reference_answer': 'ছাত্রটি বই পড়ছে।', 'student_answer': 'বইটি ছাত্রটিকে পড়ছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে শিক্ষককে প্রশ্ন করল?', 'reference_answer': 'ছাত্রটি শিক্ষককে প্রশ্ন করল।', 'student_answer': 'শিক্ষক ছাত্রটিকে প্রশ্ন করল।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে ছবি এঁকেছে?', 'reference_answer': 'শিল্পী ছবিটি এঁকেছে।', 'student_answer': 'ছবিটি শিল্পীকে এঁকেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'শিল্প'},
    {'question': 'বোলার কাকে আউট করেছে?', 'reference_answer': 'বোলার ব্যাটসম্যানকে আউট করেছে।', 'student_answer': 'ব্যাটসম্যান বোলারকে আউট করেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'খেলা'},
    {'question': 'কে গাড়ি চালাচ্ছে?', 'reference_answer': 'চালক গাড়িটি চালাচ্ছে।', 'student_answer': 'গাড়িটি চালককে চালাচ্ছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'মা কাকে বকা দিয়েছে?', 'reference_answer': 'মা ছেলেকে বকা দিয়েছে।', 'student_answer': 'ছেলে মাকে বকা দিয়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে কাকে পরাজিত করেছে?', 'reference_answer': 'আলেকজান্ডার পোরাসকে পরাজিত করেছেন।', 'student_answer': 'পোরাস আলেকজান্ডারকে পরাজিত করেছেন।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'কে কাকে কামড়েছে?', 'reference_answer': 'কুকুরটি মানুষটিকে কামড়েছে।', 'student_answer': 'মানুষটি কুকুরটিকে কামড়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে ফুল তুলেছে?', 'reference_answer': 'মেয়েটি ফুল তুলেছে।', 'student_answer': 'ফুলটি মেয়েটিকে তুলেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    # --- Negation ---
    {'question': 'সূর্য কোথায় ওঠে?', 'reference_answer': 'সূর্য পূর্বদিকে ওঠে।', 'student_answer': 'সূর্য পূর্বদিকে ওঠে না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'পৃথিবী কীসের চারদিকে ঘোরে?', 'reference_answer': 'পৃথিবী সূর্যের চারদিকে ঘোরে।', 'student_answer': 'পৃথিবী সূর্যের চারদিকে ঘোরে না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'জলীয়বাষ্প থেকে কী তৈরি হয়?', 'reference_answer': 'জলীয়বাষ্প ঘনীভূত হয়ে মেঘ তৈরি করে।', 'student_answer': 'জলীয়বাষ্প ঘনীভূত হয়ে মেঘ তৈরি করে না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'মানুষের শরীরে কয়টি হাড় থাকে?', 'reference_answer': 'মানুষের শরীরে ২০৬টি হাড় থাকে।', 'student_answer': 'মানুষের শরীরে ২০৬টি হাড় থাকে না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'গাছ কী তৈরি করে?', 'reference_answer': 'গাছ অক্সিজেন তৈরি করে।', 'student_answer': 'গাছ অক্সিজেন তৈরি করে না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'বরফ গলে কী হয়?', 'reference_answer': 'বরফ গলে জল হয়।', 'student_answer': 'বরফ গলে জল হয় না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'কে ভারতের প্রথম প্রধানমন্ত্রী ছিলেন?', 'reference_answer': 'জওহরলাল নেহরু ভারতের প্রথম প্রধানমন্ত্রী ছিলেন।', 'student_answer': 'জওহরলাল নেহরু ভারতের প্রথম প্রধানমন্ত্রী ছিলেন না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'বাংলাদেশ কবে স্বাধীন হয়েছিল?', 'reference_answer': 'বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।', 'student_answer': 'বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়নি।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'সিপাহি বিদ্রোহ কবে হয়েছিল?', 'reference_answer': 'সিপাহি বিদ্রোহ ১৮৫৭ সালে হয়েছিল।', 'student_answer': 'সিপাহি বিদ্রোহ ১৮৫৭ সালে হয়নি।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'চাঁদ কীভাবে আলো দেয়?', 'reference_answer': 'চাঁদ সূর্যের আলো প্রতিফলিত করে আলো দেয়।', 'student_answer': 'চাঁদ সূর্যের আলো প্রতিফলিত করে আলো দেয় না।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'সর্বোচ্চ পর্বতশৃঙ্গ কোনটি?', 'reference_answer': 'মাউন্ট এভারেস্ট সর্বোচ্চ পর্বতশৃঙ্গ।', 'student_answer': 'মাউন্ট এভারেস্ট সর্বোচ্চ পর্বতশৃঙ্গ নয়।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ভূগোল'},
    {'question': 'সবচেয়ে বড় গ্রহ কোনটি?', 'reference_answer': 'সৌরজগতের সবচেয়ে বড় গ্রহ বৃহস্পতি।', 'student_answer': 'সৌরজগতের সবচেয়ে বড় গ্রহ বৃহস্পতি নয়।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'শব্দের গতি কত?', 'reference_answer': 'শব্দের গতি প্রতি সেকেন্ডে ৩৪৩ মিটার।', 'student_answer': 'শব্দের গতি প্রতি সেকেন্ডে ৩৪৩ মিটার নয়।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'রক্তের রং কী?', 'reference_answer': 'রক্তের রং লাল।', 'student_answer': 'রক্তের রং লাল নয়।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'গাছের পাতার রং কী?', 'reference_answer': 'গাছের পাতার রং সবুজ।', 'student_answer': 'গাছের পাতার রং সবুজ নয়।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'কে আমেরিকা আবিষ্কার করেছিলেন?', 'reference_answer': 'ক্রিস্টোফার কলম্বাস আমেরিকা আবিষ্কার করেছিলেন।', 'student_answer': 'ক্রিস্টোফার কলম্বাস আমেরিকা আবিষ্কার করেননি।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'ভাষা আন্দোলন কবে হয়েছিল?', 'reference_answer': 'ভাষা আন্দোলন ১৯৫২ সালে হয়েছিল।', 'student_answer': 'ভাষা আন্দোলন ১৯৫২ সালে হয়নি।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    # --- Wrong Number ---
    {'question': 'মানুষের শরীরে কয়টি হাড় থাকে?', 'reference_answer': 'মানুষের শরীরে ২০৬টি হাড় থাকে।', 'student_answer': 'মানুষের শরীরে ২০৫টি হাড় থাকে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'বাংলাদেশ কবে স্বাধীন হয়েছিল?', 'reference_answer': 'বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়েছিল।', 'student_answer': 'বাংলাদেশ ১৯৭২ সালে স্বাধীন হয়েছিল।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'শব্দের গতি কত?', 'reference_answer': 'শব্দের গতি প্রতি সেকেন্ডে ৩৪৩ মিটার।', 'student_answer': 'শব্দের গতি প্রতি সেকেন্ডে ৩৩০ মিটার।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'ভাষা আন্দোলন কবে হয়েছিল?', 'reference_answer': 'ভাষা আন্দোলন ১৯৫২ সালে হয়েছিল।', 'student_answer': 'ভাষা আন্দোলন ১৯৫৩ সালে হয়েছিল।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    # --- Wrong Location ---
    {'question': 'সূর্য কোথায় ওঠে?', 'reference_answer': 'সূর্য পূর্বদিকে ওঠে।', 'student_answer': 'সূর্য পশ্চিমদিকে ওঠে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'সূর্য কোথায় অস্ত যায়?', 'reference_answer': 'সূর্য পশ্চিমদিকে অস্ত যায়।', 'student_answer': 'সূর্য পূর্বদিকে অস্ত যায়।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    # --- Multi-sentence partial ---
    {'question': 'মুক্তিযুদ্ধ সম্পর্কে বলো।', 'reference_answer': 'বাংলাদেশের মুক্তিযুদ্ধ ১৯৭১ সালে হয়েছিল। এটি নয় মাস স্থায়ী হয়েছিল।', 'student_answer': 'মুক্তিযুদ্ধ ১৯৭১ সালে হয়েছিল।', 'label': 'partially_correct', 'human_score': 50, 'subject': 'ইতিহাস'},
    # --- Voice Change ---
    {'question': 'কে রাবণকে বধ করেছিলেন?', 'reference_answer': 'রাম রাবণকে বধ করেছিলেন।', 'student_answer': 'রাবণের দ্বারা রাম নিহত হয়েছেন।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'কে বাঘ শিকার করেছিল?', 'reference_answer': 'শিকারী বাঘ শিকার করেছিল।', 'student_answer': 'বাঘের দ্বারা শিকারী নিহত হয়েছিল।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'পুলিশ কাকে ধরেছে?', 'reference_answer': 'পুলিশ চোরকে ধরেছে।', 'student_answer': 'চোরের দ্বারা পুলিশ ধৃত হয়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'বিড়াল কাকে ধরেছে?', 'reference_answer': 'বিড়াল ইঁদুর ধরেছে।', 'student_answer': 'ইঁদুরের দ্বারা বিড়াল ধরা পড়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
    {'question': 'কে সাপ মেরেছে?', 'reference_answer': 'কৃষক সাপ মেরেছে।', 'student_answer': 'সাপের দ্বারা কৃষক মারা গেছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'মা কাকে বকা দিয়েছে?', 'reference_answer': 'মা ছেলেকে বকা দিয়েছে।', 'student_answer': 'ছেলের দ্বারা মা বকা খেয়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে কাকে পরাজিত করেছে?', 'reference_answer': 'আলেকজান্ডার পোরাসকে পরাজিত করেছেন।', 'student_answer': 'পোরাসের দ্বারা আলেকজান্ডার পরাজিত হয়েছেন।', 'label': 'incorrect', 'human_score': 0, 'subject': 'ইতিহাস'},
    {'question': 'বোলার কাকে আউট করেছে?', 'reference_answer': 'বোলার ব্যাটসম্যানকে আউট করেছে।', 'student_answer': 'ব্যাটসম্যানের দ্বারা বোলার আউট হয়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'খেলা'},
    {'question': 'কে কাকে কামড়েছে?', 'reference_answer': 'কুকুরটি মানুষটিকে কামড়েছে।', 'student_answer': 'মানুষটির দ্বারা কুকুরটি কামড় খেয়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'সাধারণ জ্ঞান'},
    {'question': 'কে ইঁদুর খেয়েছে?', 'reference_answer': 'সাপটি ইঁদুর খেয়েছে।', 'student_answer': 'ইঁদুরের দ্বারা সাপটি খাওয়া হয়েছে।', 'label': 'incorrect', 'human_score': 0, 'subject': 'বিজ্ঞান'},
]


def load_and_augment_dataset(filepath: str, output_filepath: str) -> pd.DataFrame:
    """
    Load the original CSV from *filepath*, append adversarial examples,
    save to *output_filepath*, and return the augmented DataFrame.
    """
    print(f'Loading dataset from {filepath}...')
    df = pd.read_csv(filepath)

    df_adv = pd.DataFrame(_ADVERSARIAL_SAMPLES)
    start_id = df['id'].max() + 1
    df_adv['id'] = range(start_id, start_id + len(df_adv))
    df_adv = df_adv[['id', 'question', 'reference_answer', 'student_answer',
                      'label', 'human_score', 'subject']]

    df_augmented = pd.concat([df, df_adv], ignore_index=True)
    df_augmented.to_csv(output_filepath, index=False)
    print(f'Added {len(df_adv)} adversarial examples. Total samples: {len(df_augmented)}')
    return df_augmented


def find_dataset_csv(csv_filenames=None):
    """
    Search common Kaggle and local paths for the dataset CSV.
    Returns the first match or raises FileNotFoundError.

    On a local machine the datasets are expected inside:
        <project_root>/data/
    On Kaggle they are found under /kaggle/input/<dataset-name>/.
    """
    from src.config import DATA_DIR, DATASET_CSV_CANDIDATES  # avoid circular at module level

    # 1. Try the pre-configured candidate list first
    for path in DATASET_CSV_CANDIDATES:
        if os.path.exists(path):
            return path

    # 2. Fall back to explicit names supplied by the caller
    if csv_filenames is None:
        csv_filenames = [
            'dataset_single_sentence.csv',
            'dataset_medium_sentences.csv',
            'dataset_long_sentences.csv',
            'train_data.csv',
        ]

    # Search the data/ directory
    for name in csv_filenames:
        local_path = os.path.join(DATA_DIR, name)
        if os.path.exists(local_path):
            return local_path

    # Search Kaggle input tree
    if os.path.exists('/kaggle/input'):
        for name in csv_filenames:
            matches = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
            if matches:
                return matches[0]

    # Last resort: current working directory
    for name in csv_filenames:
        if os.path.exists(name):
            return name

    raise FileNotFoundError(
        f'Could not find any dataset CSV.\n'
        f'Looked in: {DATA_DIR}\n'
        f'Expected one of: {csv_filenames}\n'
        f'Please place your dataset in the data/ folder.'
    )


def split_dataset(df_augmented: pd.DataFrame, num_adversarial: int = 50,
                  test_size: float = 0.2, random_state: int = 42):
    """
    Split the augmented dataset into train and test sets.

    Adversarial samples (the last *num_adversarial* rows) are always
    placed in the test set; the normal samples are randomly split.

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    df_adv    = df_augmented.tail(num_adversarial).copy().reset_index(drop=True)
    df_normal = df_augmented.iloc[:-num_adversarial].copy().reset_index(drop=True)
    assert len(df_adv) == num_adversarial, 'Adversarial count mismatch!'

    train_df, test_normal_df = train_test_split(df_normal, test_size=test_size,
                                                random_state=random_state)
    test_df  = pd.concat([test_normal_df, df_adv]).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    return train_df, test_df
