<h1>Predicting Alumni Event Interest</h1>

<h2>Description</h2>
This project trains an XGBoost model to predict alumni interest in upcoming events using sanitized, non-identifying features. I originally built and used this approach at work to estimate likely event engagement, anticipate drop-off, and enable targeted outreach.  
<br />
The dataset in this repository is a sample for demonstration. The production dataset cannot be shared since it contains personal information. The production model achieved an accuracy above 80 percent on held-out data.
<br />

<h2>Languages and Utilities Used</h2>

- <b>Python</b>
- <b>XGBoost</b>
- <b>scikit-learn</b>
- <b>pandas</b>, <b>numpy</b>
- <b>matplotlib</b>

<h2>Environments Used</h2>

- <b>Windows 10</b>

<h2>Data Schema (Sanitized)</h2>

The training script expects columns with these sanitized names:
<ul>
<li><code>belongs_to_groups</code> (integer count, binarized in code)</li>
<li><code>belongs_to_household</code> (integer count, binarized in code)</li>
<li><code>educational_involvement</code> (integer count, binarized in code)</li>
<li><code>lifetime_raised</code> (numeric currency, symbols and commas cleaned in code)</li>
<li><code>first_degree</code>, <code>last_degree</code> (categoricals, used to create <code>multiple_degree</code>)</li>
<li><code>multiple_degree</code> (0 or 1, created if missing)</li>
<li><code>events_target_year</code> <b>(target)</b> integer count of events</li>
<li>Optional: <code>person_id</code>, <code>event_count_all</code>, <code>event_count_5years</code> are dropped from features if present</li>
</ul>

Segments are derived from <code>events_target_year</code>:
<ul>
<li>0 events → Low</li>
<li>1–2 events → Moderate</li>
<li>3–5 events → High</li>
<li>6+ events → Very High</li>
</ul>

<h2>Program walk-through:</h2>

<p align="center">
Download the files locally: <br/>
<b>EventInterestSegmentation.py</b> and <b>event_model_sample.csv</b><br/>
<br />
<br />
Install Python packages (Windows PowerShell or CMD): <br/>
<code>pip install xgboost scikit-learn pandas numpy matplotlib openpyxl</code><br/>
<br />
<br />
Put both files in the same folder. Open <code>EventInterestSegmentation.py</code> and set:  <br/>
<code>csv_file_path = r'event_model_sample.csv'</code><br/>
<br />
<br />
Run the script: <br/>
<code>python EventInterestSegmentation.py</code><br/>
<br />
<br />
Check outputs in the same folder:  <br/>
<b>constituent_segments.csv</b>, <b>constituent_segments.xlsx</b>, and <b>feature_importance.png</b><br/>
</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
