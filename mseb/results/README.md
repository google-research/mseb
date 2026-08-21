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
