function run_bench_queue(queueFile)
% RUN_BENCH_QUEUE  Process a JSON list of headless-wrapper tasks in ONE
% MATLAB session (amortizes the ~20-30 s parpool startup across all ALDVC
% tasks in the queue; the wrappers support back-to-back calls).
%
%   run_bench_queue('queue_0.json')
%
% queueFile: JSON array of task objects with fields:
%   func    : 'aldvc_headless' | 'globaldvc_headless'
%   cfg     : absolute path to the wrapper cfg JSON
%   outFile : absolute path of the expected output .mat (skip if exists)
%   sidecar : absolute path of the per-task runtime sidecar JSON to write
%
% Per task: skip-existing, RAM gate (wait while system free RAM < 6 GB,
% max 30 min), try/catch isolation, wall-clock sidecar
% {"wall_s": ..., "ok": true/false, "err": "..."}.
%
% Benchmark-v2 campaign helper (2026-07-14). Adds functionality only; no
% original ALDVC / Global_DVC sources are touched.

raw = fileread(queueFile);
tasks = jsondecode(raw);
if isempty(tasks)
    fprintf('[run_bench_queue] empty queue: %s\n', queueFile);
    return
end
n = numel(tasks);
fprintf('[run_bench_queue] %d task(s) from %s\n', n, queueFile);

for k = 1:n
    t = tasks(k);
    if exist(t.outFile, 'file')
        fprintf('[queue %d/%d] SKIP (output exists): %s\n', k, n, t.outFile);
        continue
    end

    % ---- RAM gate: require > 6 GB free physical memory ----
    % Self-healing (2026-07-14): idle parpools hold ~10-13 GB per session;
    % with several concurrent sessions the gate can livelock (nobody's
    % free RAM recovers because every session's idle pool holds it).
    % After 4 failed attempts, close THIS session's own idle pool -- the
    % next ALDVC task re-opens it on demand (~20-30 s), which is far
    % cheaper than stalling 30 min.
    for attempt = 1:60
        [~, sysview] = memory;
        if sysview.PhysicalMemory.Available > 8 * 2^30
            break
        end
        if attempt == 4 && ~isempty(gcp('nocreate'))
            fprintf('[queue %d/%d] RAM-gated: closing own idle parpool\n', k, n);
            delete(gcp('nocreate'));
            continue
        end
        fprintf('[queue %d/%d] free RAM < 8 GB, waiting 30 s (attempt %d)\n', ...
                k, n, attempt);
        pause(30);
    end

    fprintf('[queue %d/%d] %s(''%s'')\n', k, n, t.func, t.cfg);
    tStart = tic;
    ok = true;
    errmsg = '';
    try
        res = feval(t.func, t.cfg); %#ok<NASGU>
    catch ME
        ok = false;
        errmsg = ME.message;
        fprintf(2, '[queue %d/%d] ERROR: %s\n', k, n, ME.message);
    end
    wall = toc(tStart);
    ok = ok && (exist(t.outFile, 'file') ~= 0);

    sc = struct('wall_s', round(wall, 2), 'ok', ok, 'err', errmsg);
    fid = fopen(t.sidecar, 'w');
    if fid > 0
        fprintf(fid, '%s', jsonencode(sc));
        fclose(fid);
    end
    fprintf('[queue %d/%d] done: ok=%d wall=%.1f s\n', k, n, ok, wall);
end
fprintf('[run_bench_queue] queue complete: %s\n', queueFile);
end
