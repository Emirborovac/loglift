# Overnight self-labeling cycle: harvest crops from every newly downloaded
# well, then retrain the digit reader. Safe to re-run; harvest skips wells
# already in the manifest.
Set-Location "X:\Coding junk 2\Personal Projects\loglift"
python -u -m training.harvest_labels 2>&1 | Out-File -Append -Encoding utf8 data\overnight_harvest.log
python -u -m training.train_digits 2>&1 | Out-File -Append -Encoding utf8 data\overnight_train.log
"done $(Get-Date)" | Out-File -Append -Encoding utf8 data\overnight_done.txt
