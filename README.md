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
Clone and install: <br/>
<code>git clone &lt;your-repo-url&gt; && cd &lt;repo&gt; && pip install -r requirements.txt</code><br/>
<br />
<br />
Set your CSV path in <code>train_event_model.py</code>:  <br/>
<code>csv_file_path = r'path_to_your_data.csv'</code><br/>
<br />
<br />
Run training: <br/>
<code>python train_event_model.py</code><br/>
<br />
<br />
Review feature importance:  <br/>
<img src="feature_importance.png" height="80%" width="80%" alt="XGBoost Feature Importance"/>
<br />
<br />
Check outputs:  <br/>
<b>constituent_segments.csv</b>, <b>constituent_segments.xlsx</b> (predicted segments) and <b>sample_weights_by_segment.csv</b> (diagnostics).
</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
