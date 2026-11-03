# Dataset inventory

Initial inspection of the supplied `Dataset/` folder:

- 311 images across 12 denomination/side folders
- 311 JPEG files; all passed Pillow's image integrity check
- Dimensions: 180 images at 3000×4000 and 131 at 4000×3000
- Total size: approximately 2.51 GiB
- Target candidate folders: 53 `INR50_Front` and 53 `INR50_Back` images
- Background-only candidate folders: 205 images across the other denominations
- Bounding-box labels: none present at inspection time
- Exact byte-for-byte duplicates: none found in a SHA-256 comparison of all 311 files
- Corrupt or unreadable images: 0

`python -m src.analyze_dataset` verified image content, calculated SHA-256 hashes,
and wrote the per-folder results and duplicate check to
`reports/dataset_analysis.json`.

| Folder | Images |
|---|---:|
| `INR10_Back` | 21 |
| `INR10_Front` | 21 |
| `INR20_Back` | 21 |
| `INR20_Front` | 22 |
| `INR50_Back` | 53 |
| `INR50_Front` | 53 |
| `INR100_Back` | 20 |
| `INR100_Front` | 20 |
| `INR200_Back` | 20 |
| `INR200_Front` | 20 |
| `INR500_Back` | 20 |
| `INR500_Front` | 20 |

These are counts and file-health observations only, not a visual audit of every
photo. Confirm the full `INR50_*` folders contain only the intended new ₹50
design before using them as positive examples.
