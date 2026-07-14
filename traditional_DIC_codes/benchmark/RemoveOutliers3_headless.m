function [uvw,cc,RemoveOutliersList] = RemoveOutliers3_headless(uvw,cc,medianFilterThreshold)
% REMOVEOUTLIERS3_HEADLESS  Non-interactive port of ALDVC/Global_DVC RemoveOutliers3.
%
%   [uvw,cc,RemoveOutliersList] = RemoveOutliers3_headless(uvw,cc,medianFilterThreshold)
%
% Runs the universal (median) outlier test [Westerweel & Scarano 2005] with a
% fixed threshold and inpaints removed points with inpaint_nans3 (must be on
% path from ALDVC/func or Global_DVC/func). It replicates exactly the
% non-interactive code path of the original RemoveOutliers3 called with
% qDICOrNot=0, a nonzero median threshold, and zero-norm displacement bounds
% (no q-factor removal, no manual bounds, no figures, no input() prompts).
%
% The original RemoveOutliers3 in Global_DVC contains UNCONDITIONAL input()
% prompts, and the ALDVC version prompts when the threshold is 0; this private
% copy exists so batch (matlab -batch) runs can never block on stdin.
%
% cc is passed through unmodified (qDIC removal is disabled).
% RemoveOutliersList contains linear indices (into the M*N*L node grid) of
% points that failed the median test.

if nargin < 3 || isempty(medianFilterThreshold) || medianFilterThreshold == 0
    medianFilterThreshold = 2;   % default suggested by the original prompt
end

u = uvw.u; v = uvw.v; w = uvw.w;

% ---- Degenerate-grid guard (added 2026-07-14 for the paper-1 benchmark) ----
% inpaint_nans3 and the 3x3x3 median test require a true 3-D node grid;
% small volumes with large windows can produce grids with singleton (or
% dropped trailing) dimensions, e.g. 64^3 with win 24. Skipping the test
% and passing the field through unchanged is the honest behavior for a
% grid too thin to define a neighborhood.
if ndims(u) < 3 || min(size(u)) < 2
    RemoveOutliersList = zeros(0,1);
    fprintf(['****** RemoveOutliers3_headless: node grid %s too thin; ' ...
             'outlier test skipped ******\n'], mat2str(size(u)));
    return
end

try
    epsilon = 0.1;
    [~, normFluct1] = funRemoveOutliersLocal(u,epsilon);
    [~, normFluct2] = funRemoveOutliersLocal(v,epsilon);
    [~, normFluct3] = funRemoveOutliersLocal(w,epsilon);
    normFluctMag = sqrt(normFluct1.^2 + normFluct2.^2 + normFluct3.^2);

    RemoveOutliersList = find(normFluctMag > medianFilterThreshold);

    u2 = u; v2 = v; w2 = w;
    u2(RemoveOutliersList) = NaN;
    v2(RemoveOutliersList) = NaN;
    w2(RemoveOutliersList) = NaN;

    uvw.u = inpaint_nans3(u2,0);
    uvw.v = inpaint_nans3(v2,0);
    uvw.w = inpaint_nans3(w2,0);

    fprintf('****** Finish removing outliers (headless)! Removed %d/%d points ******\n', ...
            numel(RemoveOutliersList), numel(u));
catch err
    % Pass the field through unchanged rather than kill a batch run.
    uvw.u = u; uvw.v = v; uvw.w = w;
    RemoveOutliersList = zeros(0,1);
    fprintf('****** RemoveOutliers3_headless: outlier test skipped (%s) ******\n', ...
            err.message);
end
end

%% ========================================================================
% Verbatim copies of the subfunctions used by the original RemoveOutliers3
function [medianU, normFluct] = funRemoveOutliersLocal(u,epsilon)

nSize = 3*[1 1 1];
skipIdx = ceil(prod(nSize)/2);
padOption = 'replicate';

u = inpaint_nans3(double(u),0);

medianU = medFilt3Local(u,nSize,padOption,skipIdx);
fluct = u - medianU;
medianRes = medFilt3Local(abs(fluct),nSize,padOption,skipIdx);
normFluct = abs(fluct./(medianRes + epsilon));

end

%% ========================================================================
function Vr = medFilt3Local(V0,nSize, padoption, skipIdx)
% fast median filter for 3D data with extra options (copy of medFilt3).

if nargin < 4, skipIdx = 0; end
if nargin < 3, padoption = 'symmetric'; end
if nargin < 2, nSize = [3 3 3]; end

nLength = prod(nSize);
if mod(nLength,2) == 1, padSize = floor(nSize/2);
elseif mod(nLength,2) == 0, padSize = [nSize(1)/2-1,nSize(2)/2];
end

if strcmpi(padoption,'none')
    V = V0;
else
    V = (padarray(V0,padSize(1)*[1,1,1],padoption,'pre'));
    V = (padarray(V,padSize(2)*[1,1,1],padoption,'post'));
end

S = size(V);
nLength = prod(nSize)-sum(skipIdx>1);
Vn = single(zeros(S(1)-(nSize(1)-1),S(2)-(nSize(2)-1),S(3)-(nSize(3)-1),nLength));

i = cell(1,nSize(1)); j = cell(1,nSize(2)); k = cell(1,nSize(3));
for m = 1:nSize(1), i{m} = m:(S(1)-(nSize(1)-m)); end
for m = 1:nSize(2), j{m} = m:(S(2)-(nSize(2)-m)); end
for m = 1:nSize(3), k{m} = m:(S(3)-(nSize(3)-m)); end

p = 1;
for m = 1:nSize(1)
    for n = 1:nSize(2)
        for o = 1:nSize(3)
            if p ~= skipIdx || skipIdx == 0
                Vn(:,:,:,p) = V(i{m},j{n},k{o});
            end
            p = p + 1;
        end
    end
end

if skipIdx ~= 0, Vn(:,:,:,skipIdx) = []; end
Vn = sort(Vn,4);

if mod(nLength,2) == 1
    Vr = Vn(:,:,:,ceil(nLength/2));
else
    Vr = mean(cat(4,Vn(:,:,:,nLength/2),Vn(:,:,:,nLength/2+1)),4);
end

end
