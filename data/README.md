# Data

## AMI scenario meetings

`data/ami/` contains the **137 AMI scenario meetings** used in the paper. Each is a parsed, speaker-attributed transcript. In every meeting, four participants (Project Manager, Industrial Designer, User Interface, Marketing) design a remote control over a series of meetings of about 30 minutes each.

```text
data/ami/
├── train/        # 97 meetings  (Meeting64train … Meeting160train)
├── val/          # 20 meetings  (Meeting14val … Meeting33val)
├── test/         # 20 meetings  (Meeting15test … Meeting34test)
└── manifests/    # selection manifests (which source files were kept and why)
```

| Split | Meetings | Utterances | Utterances per meeting (min / mean / max) |
|---|---:|---:|---|
| train | 97 | 49,919 | 91 / 515 / 1,616 |
| val | 20 | 12,074 | 168 / 604 / 1,093 |
| test | 20 | 8,350 | 276 / 418 / 565 |
| **total** | **137** | **70,343** | 91 / 513 / 1,616 |

The splits follow QMSum. **The paper evaluates on the full 137-meeting pool.** Mechanism ablations and robustness checks use the 20 test meetings as the matched subset; see [`docs/experiments.md`](../docs/experiments.md).

To recompute these statistics, run `python scripts/analyze_ami_stats.py`.

### Provenance

The transcripts come from the AMI portion (Product domain) of **QMSum** (Zhong et al., 2021), which builds on the **AMI Meeting Corpus** (Carletta et al., 2005). We selected the QMSum meetings that contain all four AMI roles, which are exactly its AMI scenario meetings. The files in `manifests/` record this selection.

Meeting IDs follow the pattern `Meeting<index><split>`, e.g. `Meeting28test`. `<index>` is the meeting's position in the corresponding QMSum split. Utterance IDs are `Meeting<index>_<turn>`, numbered in chronological order. The utterance text is unchanged from the source, including annotation tags such as `{vocalsound}` and `{disfmarker}`.

### Format

Each `<meeting_id>_parsed.json` is a JSON list in chronological order:

```json
[
  {
    "utterance_id": "Meeting28_001",
    "speaker_id": "Project Manager",
    "utterance_content": "{vocalsound} I just forgot their name , so uh you're i sorry , I just forgot them all ."
  }
]
```

| Field | Description |
|---|---|
| `utterance_id` | Unique ID, chronologically ordered |
| `speaker_id` | Role label, consistent across the meeting; this is the identity the delegate represents |
| `utterance_content` | Utterance text |

### Using your own transcripts

Any meeting in this format can be run through the pipeline. Episodes are drawn from the last 40% of a meeting and each needs N = 20 preceding utterances, so longer meetings yield more episodes. `scripts/convert_csv_to_json.py` converts a CSV with a `Meeting` column of speaker-labelled lines (`Project Manager: Hello …`):

```bash
python scripts/convert_csv_to_json.py --input meetings.csv --output data/custom/
```

## Licensing and attribution

- The **AMI Meeting Corpus** is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The transcripts here are redistributed under the same license, with attribution to the AMI project.
- **QMSum** is the source of the parsed meeting transcripts. See the [QMSum repository](https://github.com/Yale-LILY/QMSum) for its terms.

If you use this data, please cite:

```bibtex
@inproceedings{carletta2005ami,
  title     = {The {AMI} Meeting Corpus},
  author    = {Carletta, Jean and Kraaij, Wessel and Ashby, Simone and Bourban, Sebastien and Flynn, Michael and Guillemot, Mael and Hain, Thomas and Kadlec, Jaroslav and Karaiskos, Vasilis and Kronenthal, Melissa and Lathoud, Guillaume and Lincoln, Michael and Lisowska, Agnes and Post, Wilfried and Reidsma, Dennis and Wellner, Pierre and McCowan, Iain},
  booktitle = {Proceedings of the Symposium on Annotating and Measuring Meeting Behavior},
  year      = {2005}
}

@inproceedings{zhong-etal-2021-qmsum,
  title     = {{QMS}um: A New Benchmark for Query-based Multi-domain Meeting Summarization},
  author    = {Zhong, Ming and Yin, Da and Yu, Tao and Zaidi, Ahmad and Mutuma, Mutethia and Jha, Rahul and Awadallah, Ahmed Hassan and Celikyilmaz, Asli and Liu, Yang and Qiu, Xipeng and Radev, Dragomir},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages     = {5905--5921},
  year      = {2021},
  address   = {Online},
  publisher = {Association for Computational Linguistics}
}
```
