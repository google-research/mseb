This contains the results of the sound embedding benchmark.

# Adding or updating submissions.

Evaluate encoder performance on a task:

```
run_task --task SVQClustering --encoder raw_spectrogram_25ms_10ms_mean > \
  $OUTPUT_DIR/raw_spectrogram_25ms_10ms_mean_SVQClustering.json
```

Add task results to a directory for that encoder in results:

```
partition_results --input_glob $OUTPUT_DIR/* --output_dir $RESULTS_DIR
```

Update the leaderboard.

```
flatten_results --input_glob $RESULTS_DIR/*/*.jsonl --output_file /tmp/flat.jsonl
generate_table --input_file /tmp/flat.jsonl --output_file $RESULTS_DIR/leaderboard.html
```
