# WA+ West Rapti training — participant guide

## 0 · Before the workshop (do this at least 2 days ahead)
1. A **Google account** with about **5 GB free** in Google Drive.
2. An **Earth Engine Cloud project** (free for non-commercial use):
   go to **https://code.earthengine.google.com/register** → *Use with a Cloud Project* → *Unpaid usage* →
   choose your organisation type (government / academic / non-profit) → create a project → **note the project ID**.
   Use the **same Google account** for Colab and Earth Engine (sign out of other accounts, or use a separate
   browser profile): Earth Engine saves its exports to the Drive of the account signed in to Earth Engine.

## 1 · Setup (once, ~5 minutes)
1. Open the **00_Setup** link from the trainer → **File › Save a copy in Drive**.
2. In cell ①, type your **Earth Engine project ID** and your name.
3. **Runtime › Run all** and approve the sign-in pop-ups (Drive, then Earth Engine).
4. Every step must end with ✅. Your files are now in **My Drive › west_rapti**.

## 2 · Running the labs
Open **My Drive › west_rapti › Lab1_delineate_interactive.ipynb** (double-click → *Open with Google Colaboratory*),
then **Runtime › Run all**. Run the labs **in order: 1 → 2 → 3 → 4 → 5 → 6**. Each lab saves to `west_rapti/outputs/`
and the next lab reads from there. Lab 6b is a stand-alone exercise.

## Troubleshooting

| Message | Fix |
|---|---|
| `No Earth Engine project set` | Run **00_Setup** (cell ①), then re-run the lab. |
| `… is an OLD west_rapti.py` | Run **00_Setup** again, then **Runtime › Restart session** and **Run all**. |
| Earth Engine `permission denied` / `not registered` | Finish step 0.2; check the project ID spelling in 00_Setup. |
| `User memory limit exceeded`, tile/download errors | In the lab's CONFIG lower `DL_TILE_MB` (e.g. 8 or 4) and re-run that cell. |
| Colab disconnected / runtime restarted | **Runtime › Run all** again — finished downloads are cached and reused. |
| `FileNotFoundError` for a `.nc` / `.json` input | The previous lab did not finish — run it to the end. |
| Gauge / `HydroMetData` warning | Labs 1 and 3 work without it; ask the trainer for the data for Labs 2, 4, 6. |
| Drive full | Free space (≥ 5 GB), then re-run. |
| Export says `COMPLETED` but the file is not found | Drive is still syncing — the notebook waits. If it never appears, the file went to the Drive of the account signed in to Earth Engine. |
| No irrigated cropland (class 8) in Lab 3 | Run the 🔎 diagnostic cell after the land-use map and send the trainer its four lines. |
| Lab 4 sliders cannot match the gauge | Expected — annual outflow is set by rainfall − ET, not by CF or AWC. See Lab 4 §5b. |
| The trainer released an updated notebook | Delete that notebook from My Drive › west_rapti and run 00_Setup again (it never overwrites your notebooks). |

Still stuck? Send the trainer a screenshot of the **whole** error cell.
