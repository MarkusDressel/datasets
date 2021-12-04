# coding=utf-8

import json
import os

import datasets
import gdown

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""

_DESCRIPTION = """\
https://github.com/clovaai/cord
"""
_URL = "https://drive.google.com/uc?id=1MqhTbcj-AHXOqYoeoh12aRUwIprzTJYI"


def gdrive_downloader(url, path):
    gdown.download(url, path, quiet=False)


class CordConfig(datasets.BuilderConfig):
    """BuilderConfig for CORD"""

    def __init__(self, **kwargs):
        """BuilderConfig for CORD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CordConfig, self).__init__(**kwargs)


class Cord(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        CordConfig(name="cord", version=datasets.Version(
            "1.0.0"), description="FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "roi": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['menu.cnt',
                                   'menu.discountprice',
                                   'menu.etc',
                                   'menu.itemsubtotal',
                                   'menu.nm',
                                   'menu.num',
                                   'menu.price',
                                   'menu.sub_cnt',
                                   'menu.sub_etc',
                                   'menu.sub_nm',
                                   'menu.sub_price',
                                   'menu.sub_unitprice',
                                   'menu.unitprice',
                                   'menu.vatyn',
                                   'sub_total.discount_price',
                                   'sub_total.etc',
                                   'sub_total.othersvc_price',
                                   'sub_total.service_price',
                                   'sub_total.subtotal_price',
                                   'sub_total.tax_price',
                                   'total.cashprice',
                                   'total.changeprice',
                                   'total.creditcardprice',
                                   'total.emoneyprice',
                                   'total.menuqty_cnt',
                                   'total.menutype_cnt',
                                   'total.total_etc',
                                   'total.total_price',
                                   'void_menu.nm',
                                   'void_menu.price']
                        )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/clovaai/cord",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        url_or_urls = ['https://drive.google.com/uc?id=1MqhTbcj-AHXOqYoeoh12aRUwIprzTJYI',
                       'https://drive.google.com/uc?id=1wYdp5nC9LnHQZ2FcmOoC0eClyWvcuARU']

        downloaded_file = dl_manager.extract(
            dl_manager.download_custom(url_or_urls, gdrive_downloader))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "filepaths": downloaded_file, "mode": "/CORD/train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "filepaths": downloaded_file, "mode": "/CORD/test"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "filepaths": downloaded_file, "mode": "/CORD/dev"}
            ),
        ]

    def _generate_examples(self, filepaths, mode):
        guid = -1
        for filepath in filepaths:
            filepath_folder  = filepath + mode
            logger.info("⏳ Generating examples from = %s", filepath_folder)
            ann_dir = os.path.join(filepath_folder, "json")
            if not os.path.exists(ann_dir):
                continue
            img_dir = os.path.join(filepath_folder, "image")
            for file in sorted(os.listdir(ann_dir)):
                guid +=1
                tokens = []
                bboxes = []
                ner_tags = []

                file_path = os.path.join(ann_dir, file)
                with open(file_path, "r", encoding="utf8") as f:
                    data = json.load(f)

                image_path = os.path.join(img_dir, file)
                image_path = image_path.replace("json", "png")

                if not os.path.exists(image_path):
                    other_dir_idx = int(not (filepaths.index(filepath)+2)%2)
                    image_path = image_path.replace(
                        filepath, filepaths[other_dir_idx])

                roi = data["roi"]
                if roi:
                    top_left = [roi["x1"], roi["y1"]]
                    bottom_right = [roi["x3"], roi["y3"]]
                    bottom_left = [roi["x4"], roi["y4"]]
                    top_right = [roi["x2"], roi["y2"]]
                    roi = [top_left, top_right, bottom_right, bottom_left]
                else:
                    roi = []


                for item in data["valid_line"]:
                    for word in item['words']:
                        # get word
                        txt = word['text']

                        # get bounding box
                        x1 = word['quad']['x1']
                        y1 = word['quad']['y1']
                        x3 = word['quad']['x3']
                        y3 = word['quad']['y3']

                        box = [x1, y1, x3, y3]

                        # ADDED
                        # skip empty word
                        if len(txt) < 1:
                            continue

                        tokens.append(txt)
                        bboxes.append(box)
                        ner_tags.append(item['category'])

                yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path, "roi":roi}
