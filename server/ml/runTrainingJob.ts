import { runFullBenchmarkTraining, setProgressReporter } from './trainer.ts';
import type { ModelId, TrainingProgressState } from './types.ts';

async function main() {
  const argIds = process.argv[2] ? (JSON.parse(process.argv[2]) as ModelId[]) : undefined;
  setProgressReporter((state: TrainingProgressState) => {
    process.stdout.write(`PROGRESS:${JSON.stringify(state)}\n`);
  });

  await runFullBenchmarkTraining(argIds);
  process.stdout.write('DONE\n');
}

main().catch((err) => {
  process.stderr.write(`Training job failed: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
