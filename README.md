# Classifier-Defined Toxicity in Twitch Live Chat and YouTube Comments

Original NLP course project by Davide Peluso, Giovanna Mariotto, Erik Baricevic, Giacomo Gianoglio, and Nurassyl Bolat.

The project compares toxicity and sentiment in Twitch live chat and YouTube comments across 14 languages. It applies multilingual classifiers to a course corpus of about 4.2 million messages and compares the platforms using descriptive statistics, statistical tests, and a Bayesian logistic model that accounts for message length and language.

- [Refined course report](project_report_17.pdf) contains the methods, findings, and appendices.
- [Bayesian model summary (CSV)](Code/6_Statistical_analysis/3_Hierararchical_toxicity_glmm/glm_lang_plat_len/summary.csv) and [the same summary in JSON](Code/6_Statistical_analysis/3_Hierararchical_toxicity_glmm/glm_lang_plat_len/summary.json) contain the archived coefficient estimates and diagnostics.
- The descriptive notebook includes an archived chart of channel counts by language. The two statistical test notebooks include archived aggregate tables. These outputs come from the original course runs.

## Environment

The notebooks record Python 3.11 and 3.12 kernels. Python 3.11 is a starting point for setting up the archived dependencies. The original [classifier requirements](Code/4_Sentiment_analysis/sentiment_analysis_requirements.txt) and [Twitch correction requirements](Code/%23%23%23out_of_track/twitch_sentiment_on_threads_error/requirements.txt) are included.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r Code/4_Sentiment_analysis/sentiment_analysis_requirements.txt
```

Additional packages used by the course code are listed by stage. They were outside the submitted dependency manifests.

| Stage | Additional dependencies |
| --- | --- |
| Collection | `requests`, `google-api-python-client`, and optionally `python-dotenv` for Twitch |
| Notebooks and Parquet I/O | `jupyter`, `pyarrow`, `charset-normalizer` |
| Descriptive analysis | `matplotlib`, `plotly`, `seaborn`, `nltk` with the `stopwords` corpus, `stopwordsiso`, `Unidecode` |
| Statistical analysis | `scipy`, `statsmodels`, `pymc`, `arviz`, and a NetCDF backend such as `h5netcdf` |

The classifiers use `cardiffnlp/twitter-xlm-roberta-base-sentiment` and `textdetox/xlmr-large-toxicity-classifier-v2`. The shell launchers use the course Slurm partition and Miniconda module. Configure the local environment and prepare `out/` and `err/` in the job directory before using those launchers.

## Workflow and required data

The two platform branches run in parallel through classification. For a historical rerun, place the authorized course data in the locations below. Run notebooks and launch scripts from their own stage folders where they use relative paths. Several earlier stages also use `~/Desktop/NLP_Submission/Data/` in the source. Point those paths to your private data directory before running them locally.

| Order | Course code | Inputs and outputs |
| --- | --- | --- |
| 1. Collection | `Code/1_Scraping/twitch_scraper.py` and `youtube_scraper.py` | Twitch takes `STREAMER_CSV`, defaulting to `Data/all_streamer.csv`, and uses `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, `TWITCH_NICK`, and `TWITCH_CHAT_TOKEN`. YouTube reads `Data/all_streamer.csv` under its configured project directory and needs a YouTube API key in a private local copy. The archived streamer lists have Twitch handles and language labels. The YouTube code expects `YouTube Link` and `Nick`. The merger notebooks look for raw CSV batches in `Data/1_Scraped_data/scraped_twitch/` and `Data/1_Scraped_data/scraped_youtube/`. |
| 2. Merging | `Code/2_Merging/twitch_comment_merger.ipynb` and `youtube_comment_merger.ipynb` | Reads raw CSVs and writes deduplicated records under `Data/2_Merged_data/`. The submitted Twitch scaler expects `twitch_merged/twitch_merged.csv`. The submitted YouTube scaler expects `youtube_merged/youtube_merged.csv`. |
| 3. Scaling | `Code/3_Scaling/twitch_scaler.ipynb` and `youtube_scaler.ipynb` | Samples the platform data by language toward the course's 2:1 Twitch-to-YouTube target. The archived scaled datasets are `Data/3_Scaled_data/twitch_scaled.csv` and `Data/3_Scaled_data/youtube_scaled.parquet`. |
| 4. Classification | `Code/4_Sentiment_analysis/sentiment_analysis_twitch.py` and `sentiment_analysis_youtube.py`, with their corresponding `run_sentiment_analysis_*.sh` launchers | The stage-local entry points read `filt.csv` for Twitch and `merged.csv` for YouTube. They produce `sent_chunk_*.parquet` and metadata in `sent_test_out_t/` and `sent_test_out_y/`. The archived final analysis inputs are `Data/4_Sentiment_data/twitch_sentiment.csv` and `Data/4_Sentiment_data/youtube_sentiment.csv`. |
| 5. Descriptive analysis | `Code/5_Descriptive_analysis/descriptive_analysis_twitch_youtube.ipynb` | Reads both final sentiment CSVs and produces descriptive tables and charts. |
| 6a. Statistical preparation | `preproccess_yt_tt_GLMM.ipynb`, then `preprocess_2_yt_tt_GLMM.ipynb`, both in [`Code/6_Statistical_analysis/1_Welch’s t-test/`](Code/6_Statistical_analysis/1_Welch%E2%80%99s%20t-test/) | The first reads the final sentiment CSVs and writes stage-local `twitch.csv` and `youtube.csv`. The second reads those CSVs and writes `ready_all_langs.parquet`. The archived stage CSVs are byte-identical to the corresponding final sentiment CSVs. |
| 6b. Statistical tests | `stats_coehns_t_test_final.ipynb` and `updated_stats_analysis_final.ipynb` in the same folder | The first reads `ready_all_langs.parquet` and runs Welch’s test on the binary hard-toxicity flag and proportion Z-tests with Cohen’s h. The second reads the stage-local CSVs and runs Mann–Whitney tests on classifier scores and Z-tests on binary labels. |
| 6c. Bayesian model | `run_glmm.ipynb` or `glm_lang_plat_len.py` in [`Code/6_Statistical_analysis/3_Hierararchical_toxicity_glmm/`](Code/6_Statistical_analysis/3_Hierararchical_toxicity_glmm/) | Place `ready_all_langs.parquet` beside the code. The archived copy matches the one in 6a. The run writes `glm_lang_plat_len/ready_df.parquet`, `trace.nc`, `summary.csv`, and `summary.json`. The private trace is relevant to the `--resume` option. The notebook requests 1,500 tuning steps and 1,500 draws. The CLI defaults to 2,000 of each. |

The final sentiment CSVs are the practical starting point for descriptive and statistical reruns. The archived `ready_all_langs.parquet` is enough to start the Welch notebook or the Bayesian model. Both stages require private inputs. The plot and summary files above can be read directly from the repository.

## Course corrections and gaps

[`Code/###out_of_track/`](Code/%23%23%23out_of_track/) preserves the available course correction code and the original directory names.

- `merging_the_scraped_youtube_russian_spanish_portugues/merger_spanish_portoguese_russian.ipynb` reads adjacent `youtube_comments_enhanced_*.csv` and writes `merged.csv`. `youtube_only_spanish_russinas_portugues/merge_parquet_containings_sentiment_of_youtube.ipynb` concatenates the adjacent `sent_chunk_000_2.parquet` through `sent_chunk_011_2.parquet` and writes `df_youtube_sentiment_srp.csv`. Its exploratory cells also reference the missing `sent_chunk_000.parquet` and `df_missing.csv`.
- `twitch_sentiment_on_threads_error/` contains `set_t_v0.py`, `run.sh`, and `requirements.txt`. The classifier takes the adjacent private `filt.csv`. The submission lacks standalone code for extracting and cleaning that subset and substituting its results into the final dataset.
- `youtube_before_russians_spanish_portuguese/` and `ukranians_sentiment_lost/` contained data artifacts only. The submission lacks separate code for the supplemental YouTube rescaling and classification, the Ukrainian rescaling and classification, and the final reintegration or export to the two sentiment CSVs.
- The submitted top-level directory list reverses stages 4 and 5 and gives the statistical folders an extra `hierarchical_toxicty_glmm` parent. This README uses the actual paths. `Data/1_Scraped_data/` had no raw batches. `Data/2_Merged_data/` held `merged_youtube_comments.parquet` despite the submitted README describing it as empty. Both nested merged CSVs expected by stage 3 were absent. The two main classifier inputs, `Code/4_Sentiment_analysis/filt.csv` and `Code/4_Sentiment_analysis/merged.csv`, are missing from the course submission at those paths.
- The scrapers have leading slashes in joined output paths and use `1_scraped_data` where the merger paths say `1_Scraped_data`. The YouTube scraper and the merger notebooks also need home-directory expansion in their literal paths. The YouTube merger writes `youtube_merged/twitch_merged.csv`, while its scaler looks for `youtube_merged/youtube_merged.csv`. The YouTube scaler also targets `twitch_scaled.csv` and contains a block that reads the Twitch merged data. Resolve these path and handoff issues in a private working copy before an end-to-end rerun.
- The course report describes a Welch test on raw classifier probabilities. The available Welch code uses the binary flag. The delivered score comparison notebook uses Mann–Whitney tests. A separate raw-probability Welch implementation could not be located in the submission.

## Public copy

The source of every notebook cell is preserved. Saved row previews, local paths, warnings, and other outputs were removed from the public copies. The verified static descriptive chart and aggregate statistical tables remain. The archived Twitch correction launcher omits its personal Slurm email configuration. The scrapers retain the submission's credential placeholders.

The public copy includes the report and aggregate results. Raw comments, cleaned comment text, channel and author identifiers, observation-level datasets, and the model trace stay private. The trace contains per-observation outcomes alongside posterior draws. The `.gitignore` covers the original private input locations and generated datasets, model traces, logs, and credentials.

Restore private files at their original paths under `Data/` and `Code/`. The table lists the course artifacts outside the public copy. The names in the last row are relative to `Code/###out_of_track/`.

| Location | Private files |
| --- | --- |
| `Data/` and `Code/1_Scraping/` | `all_streamer.csv` in both locations |
| `Data/2_Merged_data/` | `merged_youtube_comments.parquet` |
| `Data/3_Scaled_data/` | `twitch_scaled.csv`, `youtube_scaled.parquet` |
| `Data/4_Sentiment_data/` | `twitch_sentiment.csv`, `youtube_sentiment.csv` |
| `Code/6_Statistical_analysis/1_Welch’s t-test/` | `twitch.csv`, `youtube.csv`, `ready_all_langs.parquet` |
| `Code/6_Statistical_analysis/3_Hierararchical_toxicity_glmm/` | `ready_all_langs.parquet`, `glm_lang_plat_len/ready_df.parquet`, `glm_lang_plat_len/trace.nc` |
| `Code/###out_of_track/` | `merging_the_scraped_youtube_russian_spanish_portugues/youtube_comments_enhanced_*.csv`, `twitch_sentiment_on_threads_error/filt.csv`, `ukranians_sentiment_lost/ukranians_sentiment.csv`, `youtube_before_russians_spanish_portuguese/youtube_scaled_p1.csv`, `youtube_only_spanish_russinas_portugues/df_youtube_sentiment_srp.csv`, `youtube_only_spanish_russinas_portugues/sent_chunk_*_2.parquet` |
