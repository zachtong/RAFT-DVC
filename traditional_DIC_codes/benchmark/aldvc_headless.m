function result = aldvc_headless(cfgFile)
% ALDVC_HEADLESS  Non-interactive batch wrapper around the ALDVC pipeline.
%
%   result = aldvc_headless(cfgFile)
%
%   cfgFile: path to a JSON file (or .mat file) with fields:
%       refFile     : path to reference volume .mat (first variable = 3-D array)
%       defFile     : path to deformed  volume .mat
%       winsize     : 1x3 subset size in voxels, e.g. [16,16,16]
%       winstepsize : 1x3 subset step  in voxels, e.g. [8,8,8]
%       outFile     : path of output .mat to write
%
%   Reproduces main_ALDVC.m Sections 1-6 (displacement only, no strain,
%   no plotting, no prompts). Pinned parameters:
%       interpMethod='cubic', clusterNo=1, initFFTMethod='bigxcorr',
%       trackingMode='cumulative', Subpb2FDOrFEM='finiteDifference',
%       ICGNtol=1e-2, Subpb1ICGNMaxIterNum=50, ALVarMu=1e-3,
%       single beta = 1e-2*mean(winstepsize)^2*ALVarMu (no L-curve),
%       2 ADMM outer iterations, gridRange = full volume.
%
%   Output .mat contains:
%       U_aldvc        : 3*Nnodes x 1, ADMM/global result [u1;v1;w1;u2;...]
%       U_local        : 3*Nnodes x 1, pure local subset IC-GN result
%       coordinatesFEM : Nnodes x 3 node coordinates (MATLAB 1-based voxel units)
%       winstepsize    : 1x3
%       timing         : struct with fields integer_search_s, local_icgn_s,
%                        global_admm_s
%
% This file only ADDS functionality; original ALDVC sources are not modified.

%% ---------- Paths (absolute, no cd juggling) ----------
thisDir  = fileparts(mfilename('fullpath'));
codeRoot = fullfile(thisDir,'..','ALDVC');
warning('off','all');
% Both codebases define same-named functions (funNormalizeImg3, MeshSetUp3,
% Init3, ...). Strip any previously-added ALDVC/Global_DVC dirs so wrappers
% can be called back-to-back in one MATLAB session in any order.
local_rmPathsContaining({[filesep 'ALDVC'],[filesep 'Global_DVC']});
% addpath PREPENDS: add benchmark LAST so it ends up FIRST on the path
% (patched headless copies must shadow originals).
addpath(codeRoot);                                  % ba_interp3.mexw64
addpath(fullfile(codeRoot,'func'));
addpath(fullfile(codeRoot,'src'));
addpath(fullfile(codeRoot,'PlotFiles'));
addpath(fullfile(codeRoot,'PlotFiles','export_fig-d966721'));
addpath(fullfile(codeRoot,'func','regularizeNd'));
addpath(thisDir);                                   % benchmark first on path

%% ---------- Config ----------
cfg = local_readCfg(cfgFile);
assert(isfield(cfg,'refFile') && isfield(cfg,'defFile') && ...
       isfield(cfg,'winsize') && isfield(cfg,'winstepsize') && ...
       isfield(cfg,'outFile'), 'aldvc_headless: cfg missing required fields');

winsize     = double(cfg.winsize(:)');      assert(numel(winsize)==3);
winstepsize = double(cfg.winstepsize(:)');  assert(numel(winstepsize)==3);

%% ---------- Load volumes (first variable convention) ----------
ImgRef = local_loadVol(cfg.refFile);
ImgDef = local_loadVol(cfg.defFile);
assert(isequal(size(ImgRef),size(ImgDef)),'ref/def volume size mismatch');

%% ---------- DVC parameters (pinned, no prompts) ----------
DVCpara = struct();
DVCpara.winsize              = winsize;
DVCpara.winstepsize          = winstepsize;
DVCpara.gridRange            = struct('gridxRange',[1,size(ImgRef,1)], ...
                                      'gridyRange',[1,size(ImgRef,2)], ...
                                      'gridzRange',[1,size(ImgRef,3)]);
DVCpara.Subpb2FDOrFEM        = 'finiteDifference';
DVCpara.clusterNo            = 1;        % MUST be 1 ('case 0||1' bug: 0 falls into parfor)
DVCpara.imgSize              = size(ImgRef);
DVCpara.trackingMode         = 'cumulative';
DVCpara.initFFTMethod        = 'bigxcorr';
DVCpara.newFFTSearch         = 1;
DVCpara.DIM                  = 3;
DVCpara.interpMethod         = 'cubic';
DVCpara.displayIterOrNot     = 0;
DVCpara.Subpb1ICGNMaxIterNum = 50;
DVCpara.ICGNtol              = 1e-2;
DVCpara.ADMMtol              = 1e-2;
% Outlier-removal parameters (non-interactive):
DVCpara.qDICOrNot              = 0;
DVCpara.medianFilterThreshold  = 2;          % nonzero => no prompt (prompt default is 2)
DVCpara.uvwUpperAndLowerBounds = zeros(6,1); % non-empty & zero-norm => no prompt, no clipping

%% ---------- Normalize images (z-score over ROI), as in main Sec 2/3 ----------
[ImgNorm1,DVCpara.gridRange] = funNormalizeImg3({ImgRef},DVCpara.gridRange,'normalize');
[ImgNorm2,~]                 = funNormalizeImg3({ImgDef},DVCpara.gridRange,'normalize');
Img = cell(2,1); Img{1} = ImgNorm1{1}; Img{2} = ImgNorm2{1};
clear ImgRef ImgDef ImgNorm1 ImgNorm2;

%% ---------- Section 3: FFT integer search initial guess ----------
fprintf('--- [aldvc_headless] Section 3: integer search ---\n');
tStart = tic;
[xyz0,uvw0,cc,sizeOfFFTSearchRegion] = IntegerSearch3Multigrid(Img,DVCpara); %#ok<ASGLU>
DVCpara.sizeOfFFTSearchRegion = sizeOfFFTSearchRegion;
cc.ccThreshold = 1.25;
[uvw,cc] = RemoveOutliers3_headless(uvw0,cc,DVCpara.medianFilterThreshold);
[DVCmesh] = MeshSetUp3(xyz0,DVCpara);
U0 = Init3(uvw,DVCmesh.xyz0);
tIntegerSearch = toc(tStart);
% Image gradient placeholder (LocalICGN3/funICGN3 only use Df.imgSize)
Df = struct(); Df.imgSize = size(Img{1});

%% ---------- Section 4: Local subset IC-GN (Subproblem 1, step 1) ----------
fprintf('--- [aldvc_headless] Section 4: local IC-GN ---\n');
tStart = tic;
[USubpb1,FSubpb1,HtempPar,~,convIterPerEle1] = LocalICGN3( ...
    U0,DVCmesh.coordinatesFEM,Df,Img{1},Img{2},DVCpara,'GaussNewton',DVCpara.ICGNtol); %#ok<ASGLU>

% ------ Remove bad local points (main_ALDVC.m lines 232-243) ------
MNLgrid = size(DVCmesh.xyz0.x);
uvw = struct();
uvw.u = reshape(USubpb1(1:3:end),MNLgrid);
uvw.v = reshape(USubpb1(2:3:end),MNLgrid);
uvw.w = reshape(USubpb1(3:3:end),MNLgrid);
[uvw,~,RemoveOutliersList] = RemoveOutliers3_headless(uvw,[],DVCpara.medianFilterThreshold);
USubpb1 = [uvw.u(:),uvw.v(:),uvw.w(:)]'; USubpb1 = USubpb1(:); FSubpb1 = FSubpb1(:);
for tempi = 0:8
    FSubpb1(9*RemoveOutliersList-tempi) = nan;
    FSubpb1(9-tempi:9:end) = reshape((inpaint_nans3(reshape(FSubpb1(9-tempi:9:end),MNLgrid),1)), ...
                                     size(DVCmesh.coordinatesFEM,1),1);
end
U_local = USubpb1;   % pure local subset DVC result (U_local_ICGN in main)
tLocalICGN = toc(tStart);

%% ---------- Section 5: first global solve (finite difference) ----------
fprintf('--- [aldvc_headless] Section 5: first global solve ---\n');
tStart = tic;
DVCpara.DispFilterSize=0; DVCpara.DispFilterStd=0;
DVCpara.StrainFilterSize=0; DVCpara.StrainFilterStd=0;
for tempi = 1:3, FSubpb1 = funSmoothStrain3(FSubpb1,DVCmesh,DVCpara); end

ALVarMu  = 1e-3;
udual = 0*FSubpb1; vdual = 0*USubpb1;
ALVarBeta = 1e-2*mean(DVCpara.winstepsize)^2*ALVarMu;   % pinned single beta (no L-curve)

MNL = 1 + (DVCmesh.coordinatesFEM(end,:)-DVCmesh.coordinatesFEM(1,:))./DVCpara.winstepsize;
FDOperator3 = funDerivativeOp3(MNL(1),MNL(2),MNL(3),DVCpara.winstepsize);
Rad = [1,1,1];
[notNeumannBCInd_F,notNeumannBCInd_U] = funFDNotNeumannBCInd3(size(DVCmesh.coordinatesFEM,1),MNL,Rad);

FResidual = FSubpb1 - udual; UResidual = USubpb1 - vdual;
FResidual = FResidual(:);    UResidual = UResidual(:);
tempAMatrixSub2 = (ALVarBeta*(FDOperator3')*FDOperator3) + ALVarMu*speye(DVCpara.DIM*prod(MNL));
USubpb2temp = tempAMatrixSub2 \ (ALVarBeta*FDOperator3'*FResidual + ALVarMu*UResidual);
USubpb2 = USubpb1; USubpb2(notNeumannBCInd_U) = USubpb2temp(notNeumannBCInd_U);
FSubpb2 = FSubpb1; temp = FDOperator3*USubpb2temp; FSubpb2(notNeumannBCInd_F) = temp(notNeumannBCInd_F);

% ------ Update dual variables (finiteDifference branch) ------
udualtemp1 = (FSubpb2 - FSubpb1); udualtemp2 = udualtemp1(notNeumannBCInd_F);
vdualtemp1 = (USubpb2 - USubpb1); vdualtemp2 = vdualtemp1(notNeumannBCInd_U);
udual = zeros(DVCpara.DIM^2*prod(MNL),1); vdual = zeros(DVCpara.DIM*prod(MNL),1);
udual(notNeumannBCInd_F) = udualtemp2; vdual(notNeumannBCInd_U) = vdualtemp2;

%% ---------- Section 6: ADMM iterations (2 outer iterations) ----------
fprintf('--- [aldvc_headless] Section 6: ADMM loop ---\n');
HPar = cell(size(HtempPar,2),1);
for tempj = 1:size(HtempPar,2), HPar{tempj} = HtempPar(:,tempj); end

ALADMMIterStep = 1;
USubpb2_prev = USubpb2;
while (ALADMMIterStep < 3)   % 2 ADMM outer iterations (original: < 4)
    ALADMMIterStep = ALADMMIterStep + 1;

    % ---- Subproblem 1 (local) ----
    fprintf('***** Start step %d Subproblem1 *****\n',ALADMMIterStep);
    [USubpb1,~,~,convIterPerEleK] = Subpb13(USubpb2,FSubpb2,udual,vdual,DVCmesh.coordinatesFEM, ...
        Df,Img{1},Img{2},ALVarMu,ALVarBeta,HPar,ALADMMIterStep,DVCpara,'GaussNewton',DVCpara.ICGNtol); %#ok<ASGLU>
    FSubpb1 = FSubpb2;

    % ---- Subproblem 2 (global, finite difference) ----
    fprintf('***** Start step %d Subproblem2 *****\n',ALADMMIterStep);
    FResidual = FSubpb1 - udual; UResidual = USubpb1 - vdual;
    USubpb2temp = tempAMatrixSub2 \ (ALVarBeta*FDOperator3'*FResidual(:) + ALVarMu*UResidual(:));
    USubpb2 = USubpb1; USubpb2(notNeumannBCInd_U) = USubpb2temp(notNeumannBCInd_U);
    FSubpb2 = FSubpb1; temp = FDOperator3*USubpb2temp; FSubpb2(notNeumannBCInd_F) = temp(notNeumannBCInd_F);

    % ---- Convergence check ----
    Update_dispU_Subpb2 = norm(USubpb2_prev - USubpb2,2)/sqrt(numel(USubpb2));
    fprintf('Updated [U] from the global step = %g\n',Update_dispU_Subpb2);

    % ---- Update dual variables ----
    udualtemp1 = (FSubpb2(:) - FSubpb1(:));
    vdualtemp1 = (USubpb2(:) - USubpb1(:));
    udual(notNeumannBCInd_F) = udual(notNeumannBCInd_F) + udualtemp1(notNeumannBCInd_F);
    vdual(notNeumannBCInd_U) = vdual(notNeumannBCInd_U) + vdualtemp1(notNeumannBCInd_U);

    USubpb2_prev = USubpb2;
    if Update_dispU_Subpb2 < DVCpara.ADMMtol, break; end
end
tGlobalADMM = toc(tStart);

%% ---------- Save results ----------
U_aldvc        = full(USubpb2);
U_local        = full(U_local);
coordinatesFEM = DVCmesh.coordinatesFEM;
winstepsize    = DVCpara.winstepsize; %#ok<NASGU>
timing = struct('integer_search_s',tIntegerSearch, ...
                'local_icgn_s',    tLocalICGN, ...
                'global_admm_s',   tGlobalADMM); %#ok<NASGU>
outDir = fileparts(cfg.outFile);
if ~isempty(outDir) && ~exist(outDir,'dir'), mkdir(outDir); end
save(cfg.outFile,'U_aldvc','U_local','coordinatesFEM','winstepsize','timing','-v7');
fprintf('--- [aldvc_headless] Saved: %s ---\n',cfg.outFile);
fprintf('    timings [s]: integer=%.2f  local=%.2f  global+ADMM=%.2f\n', ...
        tIntegerSearch,tLocalICGN,tGlobalADMM);

result = struct('U_aldvc',U_aldvc,'U_local',U_local, ...
                'coordinatesFEM',coordinatesFEM,'timing',timing);
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
