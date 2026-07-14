function result = globaldvc_headless(cfgFile)
% GLOBALDVC_HEADLESS  Non-interactive batch wrapper around FE-Global-DVC.
%
%   result = globaldvc_headless(cfgFile)
%
%   cfgFile: path to a JSON file (or .mat file) with fields:
%       refFile     : path to reference volume .mat (first variable = 3-D array)
%       defFile     : path to deformed  volume .mat
%       winstepsize : 1x3 finite element size in voxels, e.g. [8,8,8]
%       alpha       : regularization coefficient (|grad u|^2), e.g. 80
%       outFile     : path of output .mat to write
%
%   Reproduces main_FE_GlobalDVC.m Sections 1-4 (displacement only, no strain,
%   no plotting, no prompts). Pinned parameters:
%       winsize = winstepsize + [6,6,6] (ReadImage3 convention),
%       alphaList = [alpha]  (kills the main-script line-126 override),
%       maxIter = 20, tol = 1e-2, GaussPtOrder = 2, ClusterNo = 1,
%       InitFFTMethod = 'bigxcorr', gridRange = full volume.
%
%   Output .mat contains:
%       U_global       : 3*Nnodes x 1 displacement [u1;v1;w1;u2;...]
%       coordinatesFEM : Nnodes x 3 node coordinates (MATLAB 1-based voxels)
%       winstepsize    : 1x3
%       timing         : struct with fields integer_search_s, img_gradient_s,
%                        global_icgn_s
%
% This file only ADDS functionality; original Global_DVC sources are not modified.

%% ---------- Paths (absolute, no cd juggling) ----------
thisDir  = fileparts(mfilename('fullpath'));
codeRoot = fullfile(thisDir,'..','Global_DVC');
warning('off','all');
% Both codebases define same-named functions (funNormalizeImg3, MeshSetUp3,
% Init3, ...). Strip any previously-added ALDVC/Global_DVC dirs so wrappers
% can be called back-to-back in one MATLAB session in any order.
local_rmPathsContaining({[filesep 'ALDVC'],[filesep 'Global_DVC']});
% addpath PREPENDS: add benchmark LAST so it ends up FIRST on the path.
% benchmark\ shadows Global_DVC's ba_interp3.mexw64, whose DLL fails to load
% on this machine; benchmark carries a copy of the working ALDVC build.
addpath(codeRoot);
addpath(fullfile(codeRoot,'func'));
addpath(fullfile(codeRoot,'src'));
addpath(fullfile(codeRoot,'PlotFiles'));
addpath(fullfile(codeRoot,'func','regularizeNd'));
addpath(thisDir);                                   % benchmark first on path

%% ---------- Config ----------
cfg = local_readCfg(cfgFile);
assert(isfield(cfg,'refFile') && isfield(cfg,'defFile') && ...
       isfield(cfg,'winstepsize') && isfield(cfg,'alpha') && ...
       isfield(cfg,'outFile'), 'globaldvc_headless: cfg missing required fields');

winstepsize = double(cfg.winstepsize(:)');  assert(numel(winstepsize)==3);
winsize     = winstepsize + [6,6,6];        % ReadImage3.m line 111 convention
alpha       = double(cfg.alpha);

%% ---------- Load volumes (first variable convention) ----------
ImgRef = local_loadVol(cfg.refFile);
ImgDef = local_loadVol(cfg.defFile);
assert(isequal(size(ImgRef),size(ImgDef)),'ref/def volume size mismatch');

%% ---------- DVC parameters (pinned, no prompts) ----------
DVCpara = struct();
DVCpara.winsize          = winsize;
DVCpara.winstepsize      = winstepsize;
DVCpara.gridRange        = struct('gridxRange',[1,size(ImgRef,1)], ...
                                  'gridyRange',[1,size(ImgRef,2)], ...
                                  'gridzRange',[1,size(ImgRef,3)]);
DVCpara.ClusterNo        = 1;
DVCpara.InitFFTMethod    = 'bigxcorr';
DVCpara.NewFFTSearch     = 1;
DVCpara.interpmethod     = 'cubic';
DVCpara.displayIterOrNot = 0;
DVCpara.maxIter          = 20;      % original default 100
DVCpara.tol              = 1e-2;    % original default 1e-3
DVCpara.alpha            = alpha;
DVCpara.GaussPtOrder     = 2;
DVCpara.ImgSeqIncUnit    = 3;       % two-frame run; only used for bookkeeping
DVCpara.imgSize          = size(ImgRef);
DVCpara.qDICOrNot        = 0;
DVCpara.Thr0             = 2;       % median test threshold (nonzero => no prompt)

%% ---------- Normalize images ----------
[ImgNorm1,DVCpara.gridRange] = funNormalizeImg3({ImgRef},DVCpara.gridRange,'Normalize');
[ImgNorm2,~]                 = funNormalizeImg3({ImgDef},DVCpara.gridRange,'Normalize');
Img = cell(2,1); Img{1} = ImgNorm1{1}; Img{2} = ImgNorm2{1};
clear ImgRef ImgDef ImgNorm1 ImgNorm2;

%% ---------- Section 3: FFT integer search initial guess ----------
fprintf('--- [globaldvc_headless] Section 3: integer search ---\n');
tStart = tic;
[xyz0,uvw0,cc,SizeOfFFTSearchRegion] = IntegerSearch3Mg(Img,DVCpara); %#ok<ASGLU>
cc.ccThreshold = 1.25;
% Patched non-interactive outlier removal (Global's RemoveOutliers3 has
% unconditional input() prompts and cannot run headless):
[uvw,cc] = RemoveOutliers3_headless(uvw0,cc,DVCpara.Thr0);
[DVCmesh] = MeshSetUp3(xyz0,DVCpara);
U0 = Init3(uvw,DVCmesh.xyz0);
tIntegerSearch = toc(tStart);

%% ---------- Image gradients ----------
fprintf('--- [globaldvc_headless] computing image gradients (stencil7) ---\n');
tStart = tic;
Df = funImgGradient3(Img{1},'stencil7');
Df.imgSize = size(Img{1});
tImgGradient = toc(tStart);

%% ---------- Section 4: FE-based global DVC (pinned single alpha) ----------
fprintf('--- [globaldvc_headless] Section 4: global IC-GN, alpha=%g ---\n',alpha);
tStart = tic;
% alphaList = [alpha]  -- pinned; the original main line 126 unconditionally
% overrides alphaList with an L-curve sweep. We run one alpha only.
[U,normOfW,timeICGN] = funGlobalICGN3(DVCmesh,Df,Img{1},Img{2},U0,alpha, ...
                                      DVCpara.tol,DVCpara.maxIter);
tGlobalICGN = toc(tStart);

%% ---------- Save results ----------
U_global       = full(U);
coordinatesFEM = DVCmesh.coordinatesFEM;
winstepsize    = DVCpara.winstepsize; %#ok<NASGU>
timing = struct('integer_search_s',tIntegerSearch, ...
                'img_gradient_s',  tImgGradient, ...
                'global_icgn_s',   tGlobalICGN); %#ok<NASGU>
normOfW = full(normOfW); timeICGN = full(timeICGN); %#ok<NASGU>
outDir = fileparts(cfg.outFile);
if ~isempty(outDir) && ~exist(outDir,'dir'), mkdir(outDir); end
save(cfg.outFile,'U_global','coordinatesFEM','winstepsize','timing', ...
     'normOfW','timeICGN','-v7');
fprintf('--- [globaldvc_headless] Saved: %s ---\n',cfg.outFile);
fprintf('    timings [s]: integer=%.2f  gradient=%.2f  globalICGN=%.2f\n', ...
        tIntegerSearch,tImgGradient,tGlobalICGN);

result = struct('U_global',U_global,'coordinatesFEM',coordinatesFEM,'timing',timing);
end

%% ======================================================================
function local_rmPathsContaining(patterns)
p = strsplit(path, pathsep);
for k = 1:numel(p)
    for j = 1:numel(patterns)
        if contains(p{k}, patterns{j})
            rmpath(p{k});
            break
        end
    end
end
end

%% ======================================================================
function cfg = local_readCfg(cfgFile)
[~,~,ext] = fileparts(cfgFile);
if strcmpi(ext,'.mat')
    cfg = load(cfgFile);
    if isfield(cfg,'cfg'), cfg = cfg.cfg; end
else
    fid = fopen(cfgFile,'r');
    assert(fid>0,'Cannot open cfg file: %s',cfgFile);
    raw = fread(fid,inf,'*char')'; fclose(fid);
    cfg = jsondecode(raw);
end
end

%% ======================================================================
function I = local_loadVol(fileName)
% First-variable-in-file convention (cell {1} or bare 3-D array), cast double.
S = load(fileName);
fn = fieldnames(S);
I = S.(fn{1});
if iscell(I), I = I{1}; end
I = double(I);
assert(ndims(I)==3,'Volume in %s is not 3-D',fileName);
end
