function validate_optimizations(mode)
% VALIDATE_OPTIMIZATIONS  Equivalence + timing gate for benchmark shadow-copy
% optimizations (waitbar strip / ALDVC parfor / vectorized global assembly).
%
%   validate_optimizations('ref')
%       Run BEFORE the shadow copies of the hot functions exist: executes the
%       pristine original path (serial, waitbars intact) on the 64^3 smoke
%       case and the 128^3 S6 sample. Stores reference outputs in
%       data/val/ref_*.mat and wall-clock in data/val/summary_ref.mat.
%       (Once shadow copies exist in benchmark\ this mode is no longer
%       "pristine" -- benchmark\ shadows the originals by path order.)
%
%   validate_optimizations('opt')   [default]
%       Run AFTER the shadow copies exist. For each case & method it runs
%       (a) the serial shadow path  (clusterNo=1, fast=false)  -> data/val/serial_*
%       (b) the optimized path      (clusterNo=12, fast=true)  -> data/val/opt_*
%       then compares both against the stored pristine references and prints
%       the before/after table. Errors if any max-abs displacement diff
%       exceeds HARD_TOL = 1e-6; diffs in (1e-9, 1e-6] are reported as
%       reduction-order noise; <= 1e-9 passes the strict gate.
%
% Determinism note: the ALDVC integer search injects rand() noise
% (funIntegerSearch3Multigrid 'xcorr' branch, lines ~91-92), so rng('default')
% is reset before EVERY wrapper call to make in-session runs bit-comparable
% with fresh `matlab -batch` sessions (which start at the default seed).

if nargin < 1, mode = 'opt'; end
thisDir = fileparts(mfilename('fullpath'));
cd(thisDir);
valDir = fullfile(thisDir,'data','val');  if ~exist(valDir,'dir'),  mkdir(valDir);  end
cfgDir = fullfile(thisDir,'cfgs','val');  if ~exist(cfgDir,'dir'),  mkdir(cfgDir);  end

HARD_TOL = 1e-6;   % hard equivalence bound (fail above this)
SOFT_TOL = 1e-9;   % strict gate (report-and-justify between SOFT and HARD)

caseNames  = {'smoke64','s6_128'};
aldvcCfgs  = {fullfile(thisDir,'cfgs','smoke_aldvc.json'), ...
              fullfile(thisDir,'cfgs','bench','S6','sample_00000_aldvc.json')};
globalCfgs = {fullfile(thisDir,'cfgs','smoke_global.json'), ...
              fullfile(thisDir,'cfgs','bench','S6','sample_00000_global.json')};

switch lower(mode)
    %% ------------------------------------------------------------------
    case 'ref'
        summary = struct();
        for c = 1:numel(caseNames)
            summary.(caseNames{c}).aldvc  = local_runOne(aldvcCfgs{c}, 'aldvc', caseNames{c},'ref',struct(),valDir,cfgDir);
            summary.(caseNames{c}).global = local_runOne(globalCfgs{c},'global',caseNames{c},'ref',struct(),valDir,cfgDir);
        end
        save(fullfile(valDir,'summary_ref.mat'),'summary');
        fprintf('\n===== REF (pristine original path) wall-clock =====\n');
        local_printWall(summary,caseNames);

    %% ------------------------------------------------------------------
    case 'opt'
        % Open the persistent pool up-front so its startup cost does not
        % pollute the first timed optimized run.
        if isempty(gcp('nocreate'))
            fprintf('--- opening parpool(Processes,12) ...\n');
            parpool('Processes',12);
        end

        refS = load(fullfile(valDir,'summary_ref.mat')); refSummary = refS.summary;
        optSummary = struct(); serSummary = struct();
        for c = 1:numel(caseNames)
            cn = caseNames{c};
            serSummary.(cn).aldvc  = local_runOne(aldvcCfgs{c}, 'aldvc', cn,'serial',struct('clusterNo',1),           valDir,cfgDir);
            serSummary.(cn).global = local_runOne(globalCfgs{c},'global',cn,'serial',struct('fast',false),            valDir,cfgDir);
            optSummary.(cn).aldvc  = local_runOne(aldvcCfgs{c}, 'aldvc', cn,'opt',   struct('clusterNo',12),          valDir,cfgDir);
            optSummary.(cn).global = local_runOne(globalCfgs{c},'global',cn,'opt',   struct('fast',true),             valDir,cfgDir);
        end
        save(fullfile(valDir,'summary_opt.mat'),'optSummary','serSummary');

        % --- Which functions were actually resolved (after the last run) ---
        fprintf('\n===== which() diagnostics (path as left by last globaldvc run) =====\n');
        for f = {'funGlobalICGN3','funGlobalICGN3_fast','funIntegerSearch3Mg','ba_interp3'}
            fprintf('  %-26s -> %s\n', f{1}, which(f{1}));
        end

        % --- Numerical comparison ---
        fprintf('\n===== Numerical equivalence vs pristine reference =====\n');
        allPass = true; rows = {};
        for c = 1:numel(caseNames)
            cn = caseNames{c};
            % ALDVC: U_local and U_aldvc
            ref = load(fullfile(valDir,sprintf('ref_%s_aldvc.mat',cn)));
            ser = load(fullfile(valDir,sprintf('serial_%s_aldvc.mat',cn)));
            opt = load(fullfile(valDir,sprintf('opt_%s_aldvc.mat',cn)));
            dSer_local = max(abs(ser.U_local - ref.U_local));
            dSer_aldvc = max(abs(ser.U_aldvc - ref.U_aldvc));
            dOpt_local = max(abs(opt.U_local - ref.U_local));
            dOpt_aldvc = max(abs(opt.U_aldvc - ref.U_aldvc));
            rows(end+1,:) = {cn,'U_local ',dSer_local,dOpt_local}; %#ok<AGROW>
            rows(end+1,:) = {cn,'U_aldvc ',dSer_aldvc,dOpt_aldvc}; %#ok<AGROW>
            allPass = allPass && dOpt_local < HARD_TOL && dOpt_aldvc < HARD_TOL;
            % Global: U_global (+ iteration count sanity)
            ref = load(fullfile(valDir,sprintf('ref_%s_global.mat',cn)));
            ser = load(fullfile(valDir,sprintf('serial_%s_global.mat',cn)));
            opt = load(fullfile(valDir,sprintf('opt_%s_global.mat',cn)));
            dSer_glob = max(abs(ser.U_global - ref.U_global));
            dOpt_glob = max(abs(opt.U_global - ref.U_global));
            rows(end+1,:) = {cn,'U_global',dSer_glob,dOpt_glob}; %#ok<AGROW>
            allPass = allPass && dOpt_glob < HARD_TOL;
            fprintf('  [%s] global ICGN iterations: ref=%d serial=%d fast=%d\n', ...
                cn, numel(ref.normOfW), numel(ser.normOfW), numel(opt.normOfW));
        end

        fprintf('\n  %-8s %-9s %-22s %-22s %s\n','case','field','serial-vs-ref maxdiff','opt-vs-ref maxdiff','verdict');
        for r = 1:size(rows,1)
            d = rows{r,4};
            if d <= SOFT_TOL, verdict = 'PASS (<1e-9)';
            elseif d <= HARD_TOL, verdict = 'PASS* (reduction-order noise 1e-9..1e-6)';
            else, verdict = 'FAIL (>1e-6)';
            end
            fprintf('  %-8s %-9s %-22.3e %-22.3e %s\n',rows{r,1},rows{r,2},rows{r,3},rows{r,4},verdict);
        end

        % --- Timing table ---
        fprintf('\n===== Wall-clock (s): pristine ref | serial shadow | optimized =====\n');
        fprintf('  %-8s %-8s %10s %14s %12s %9s\n','case','method','ref','serial-shadow','optimized','speedup');
        for c = 1:numel(caseNames)
            cn = caseNames{c};
            for m = {'aldvc','global'}
                tr = refSummary.(cn).(m{1}).wall;
                ts = serSummary.(cn).(m{1}).wall;
                to = optSummary.(cn).(m{1}).wall;
                fprintf('  %-8s %-8s %10.1f %14.1f %12.1f %8.1fx\n',cn,m{1},tr,ts,to,tr/to);
            end
        end

        if ~allPass
            error('validate_optimizations: EQUIVALENCE GATE FAILED (see table above).');
        end
        fprintf('\nvalidate_optimizations: ALL EQUIVALENCE CHECKS PASSED.\n');

    otherwise
        error('Unknown mode "%s" (use ''ref'' or ''opt'').',mode);
end
end

%% ======================================================================
function T = local_runOne(baseCfgFile, method, caseName, tag, extra, valDir, cfgDir)
% Run one wrapper call with overridden outFile (+extra cfg fields); return timing.
raw = fileread(baseCfgFile);
cfg = jsondecode(raw);
cfg.outFile = fullfile(valDir, sprintf('%s_%s_%s.mat',tag,caseName,method));
fn = fieldnames(extra);
for k = 1:numel(fn), cfg.(fn{k}) = extra.(fn{k}); end
cfgFile = fullfile(cfgDir, sprintf('%s_%s_%s.mat',tag,caseName,method));
save(cfgFile,'cfg');
fprintf('\n########## [%s] %s / %s ##########\n',tag,caseName,method);
rng('default');   % determinism: integer search injects rand() noise
t0 = tic;
if strcmp(method,'aldvc')
    result = aldvc_headless(cfgFile);
else
    result = globaldvc_headless(cfgFile);
end
T = struct('wall',toc(t0),'stages',result.timing);
fprintf('>>> [%s] %s / %s wall-clock = %.1f s\n',tag,caseName,method,T.wall);
end

%% ======================================================================
function local_printWall(summary,caseNames)
for c = 1:numel(caseNames)
    cn = caseNames{c};
    fprintf('  %-8s aldvc  %8.1f s   global %8.1f s\n',cn, ...
        summary.(cn).aldvc.wall, summary.(cn).global.wall);
end
end
