function [U,normOfW,timeICGN] = funGlobalICGN3_fast(DVCmesh,Df,Img1,Img2,U,alpha,tol,maxIter)
% FUNGLOBALICGN3_FAST  Vectorized drop-in replacement for funGlobalICGN3.
% =========================================================
% [BENCHMARK OPTIMIZED COPY] based on Global_DVC\func\funGlobalICGN3.m
%
% Same inputs/outputs and same IC-GN iteration logic as the original.
% Performance changes (numerics preserved to reduction-order rounding):
%   1) The interpreted per-voxel inner loop (N / DN / J rebuilt per voxel per
%      element per iteration) is replaced by:
%        - per-voxel shape-function matrix NT8 (8 x P) precomputed once;
%        - the regularizer contribution AregSum = sum_p alpha*DN_p'*DN_p
%          precomputed ONCE from element 1 (valid because the mesh is a
%          uniform axis-aligned brick grid -- verified below; the Jacobian is
%          translation-invariant so DN is identical for every element);
%        - image term assembled with matrix products:
%            tempA = B*B' + AregSum          (iteration 1 only, as original)
%            tempb = B*res - AregSum*U_ele   (every iteration)
%          where B(3(k-1)+c, p) = N_k(p)*Df_c(p) and res = f - g(x+u).
%   2) ba_interp3 is called ONCE per chunk of elements (batched query points)
%      instead of once per element. ba_interp3 evaluates each query point
%      independently, so batching is bitwise-identical.
%   3) Sparse triplets are preallocated (the original grew INDEXAI/INDEXBI by
%      concatenation inside the element loop). Triplet ordering (element-major)
%      matches the original exactly, so sparse() duplicate summation order is
%      unchanged.
%   4) waitbars removed.
%
% Numerical equivalence: identical query points (bitwise) for the warped
% volume interpolation; matrix-product summation order differs from the
% original's sequential per-voxel accumulation, so A and b entries agree only
% to relative rounding error (~1e-15). Validated end-to-end by
% benchmark\validate_optimizations.m (max |U_fast - U_original| gate).
%
% If the mesh is NOT a uniform brick grid this function falls back to the
% original funGlobalICGN3 (benchmark shadow copy, waitbar-stripped).
% =========================================================

coordinatesFEM = DVCmesh.coordinatesFEM; % FE-mesh coordinates
elementsFEM = DVCmesh.elementsFEM;       % FE-mesh elements
DIM = 3;                                 % Problem dimension
NodesPerEle = 8;                         % Using cubic elements
FEMSize = DIM*size(coordinatesFEM,1);    % FE-system size
winsize = (coordinatesFEM(2,1)-coordinatesFEM(1,1))*ones(1,3); % Finite element size

DfDx = Df.DfDx; DfDy = Df.DfDy; DfDz = Df.DfDz; % ROI image gradients
DfAxis = Df.DfAxis; DfDxStartx = DfAxis(1); DfDxStarty = DfAxis(3); DfDxStartz = DfAxis(5);
if nargin < 8 || isempty(maxIter), maxIter = 100; end

NE = size(elementsFEM,1);

%% ---------- Verify uniform axis-aligned brick mesh (else fall back) ----------
pt1All = coordinatesFEM(elementsFEM(:,1),:);                              % NE x 3
off1 = coordinatesFEM(elementsFEM(1,:),:) - repmat(pt1All(1,:),8,1);      % 8 x 3
meshOK = isequal(off1(7,:), winsize);
if meshOK
    for k = 2:8
        ptk = coordinatesFEM(elementsFEM(:,k),:);
        if any(any(ptk ~= pt1All + repmat(off1(k,:),NE,1)))
            meshOK = false; break;
        end
    end
end
if ~meshOK
    warning('funGlobalICGN3_fast:fallback', ...
        'Non-uniform mesh detected; falling back to original funGlobalICGN3.');
    [U,normOfW,timeICGN] = funGlobalICGN3(DVCmesh,Df,Img1,Img2,U,alpha,tol,maxIter);
    return
end

%% ---------- Per-voxel precomputations (identical formulas to original) ----------
ksiList = -1:2/winsize(1):1; etaList = -1:2/winsize(2):1; zetaList = -1:2/winsize(3):1;
[ksiMat,etaMat,zetaMat] = ndgrid(ksiList,etaList,zetaList);

NMat = cell(8,1);
NMat{1} = 1/8*(1-ksiMat).*(1-etaMat).*(1-zetaMat); NMat{2} = 1/8*(1+ksiMat).*(1-etaMat).*(1-zetaMat);
NMat{3} = 1/8*(1+ksiMat).*(1+etaMat).*(1-zetaMat); NMat{4} = 1/8*(1-ksiMat).*(1+etaMat).*(1-zetaMat);
NMat{5} = 1/8*(1-ksiMat).*(1-etaMat).*(1+zetaMat); NMat{6} = 1/8*(1+ksiMat).*(1-etaMat).*(1+zetaMat);
NMat{7} = 1/8*(1+ksiMat).*(1+etaMat).*(1+zetaMat); NMat{8} = 1/8*(1-ksiMat).*(1+etaMat).*(1+zetaMat);

P = numel(ksiMat);                       % voxels per element
NT8 = zeros(8,P);
for k = 1:8, NT8(k,:) = reshape(NMat{k},1,P); end
REPNT = repelem(NT8,3,1);                % 24 x P : row 3(k-1)+c = N_k

% --- Regularizer block: AregSum = sum_p (alpha*DN_p')*DN_p (element 1 corners) ---
pts = coordinatesFEM(elementsFEM(1,:),:);   % 8 x 3 corner coordinates
pt1x = pts(1,1); pt1y = pts(1,2); pt1z = pts(1,3); pt2x = pts(2,1); pt2y = pts(2,2); pt2z = pts(2,3);
pt3x = pts(3,1); pt3y = pts(3,2); pt3z = pts(3,3); pt4x = pts(4,1); pt4y = pts(4,2); pt4z = pts(4,3);
pt5x = pts(5,1); pt5y = pts(5,2); pt5z = pts(5,3); pt6x = pts(6,1); pt6y = pts(6,2); pt6z = pts(6,3);
pt7x = pts(7,1); pt7y = pts(7,2); pt7z = pts(7,3); pt8x = pts(8,1); pt8y = pts(8,2); pt8z = pts(8,3);

AregSum = zeros(DIM*NodesPerEle,DIM*NodesPerEle);
for tempjj = 1:P
    ksi = ksiMat(tempjj); eta = etaMat(tempjj); zeta = zetaMat(tempjj);
    % ------ Build J matrix (identical to original) ------
    J = [funDN1Dksi(ksi,eta,zeta),funDN2Dksi(ksi,eta,zeta),funDN3Dksi(ksi,eta,zeta),funDN4Dksi(ksi,eta,zeta), ...
        funDN5Dksi(ksi,eta,zeta),funDN6Dksi(ksi,eta,zeta),funDN7Dksi(ksi,eta,zeta),funDN8Dksi(ksi,eta,zeta);
        funDN1Deta(ksi,eta,zeta),funDN2Deta(ksi,eta,zeta),funDN3Deta(ksi,eta,zeta),funDN4Deta(ksi,eta,zeta), ...
        funDN5Deta(ksi,eta,zeta),funDN6Deta(ksi,eta,zeta),funDN7Deta(ksi,eta,zeta),funDN8Deta(ksi,eta,zeta);
        funDN1Dzeta(ksi,eta,zeta),funDN2Dzeta(ksi,eta,zeta),funDN3Dzeta(ksi,eta,zeta),funDN4Dzeta(ksi,eta,zeta), ...
        funDN5Dzeta(ksi,eta,zeta),funDN6Dzeta(ksi,eta,zeta),funDN7Dzeta(ksi,eta,zeta),funDN8Dzeta(ksi,eta,zeta)] * ...
        [pt1x,pt1y,pt1z;pt2x,pt2y,pt2z;pt3x,pt3y,pt3z;pt4x,pt4y,pt4z;pt5x,pt5y,pt5z;pt6x,pt6y,pt6z;pt7x,pt7y,pt7z;pt8x,pt8y,pt8z];
    InvJ = inv(J); %#ok<MINV>
    % ------ Compute [DN] matrix (identical to original) ------
    DN = [InvJ zeros(3,3) zeros(3,3); zeros(3,3) InvJ zeros(3,3); zeros(3,3) zeros(3,3) InvJ] * ...
        [funDN1Dksi(ksi,eta,zeta) 0 0 funDN2Dksi(ksi,eta,zeta) 0 0 funDN3Dksi(ksi,eta,zeta) 0 0 funDN4Dksi(ksi,eta,zeta) 0 0 ...
        funDN5Dksi(ksi,eta,zeta) 0 0 funDN6Dksi(ksi,eta,zeta) 0 0 funDN7Dksi(ksi,eta,zeta) 0 0 funDN8Dksi(ksi,eta,zeta) 0 0;
        funDN1Deta(ksi,eta,zeta) 0 0 funDN2Deta(ksi,eta,zeta) 0 0 funDN3Deta(ksi,eta,zeta) 0 0 funDN4Deta(ksi,eta,zeta) 0 0 ...
        funDN5Deta(ksi,eta,zeta) 0 0 funDN6Deta(ksi,eta,zeta) 0 0 funDN7Deta(ksi,eta,zeta) 0 0 funDN8Deta(ksi,eta,zeta) 0 0;
        funDN1Dzeta(ksi,eta,zeta) 0 0 funDN2Dzeta(ksi,eta,zeta) 0 0 funDN3Dzeta(ksi,eta,zeta) 0 0 funDN4Dzeta(ksi,eta,zeta) 0 0 ...
        funDN5Dzeta(ksi,eta,zeta) 0 0 funDN6Dzeta(ksi,eta,zeta) 0 0 funDN7Dzeta(ksi,eta,zeta) 0 0 funDN8Dzeta(ksi,eta,zeta) 0 0;
        0 funDN1Dksi(ksi,eta,zeta) 0 0 funDN2Dksi(ksi,eta,zeta) 0 0 funDN3Dksi(ksi,eta,zeta) 0 0 funDN4Dksi(ksi,eta,zeta) 0 ...
        0 funDN5Dksi(ksi,eta,zeta) 0 0 funDN6Dksi(ksi,eta,zeta) 0 0 funDN7Dksi(ksi,eta,zeta) 0 0 funDN8Dksi(ksi,eta,zeta) 0 ;
        0 funDN1Deta(ksi,eta,zeta) 0 0 funDN2Deta(ksi,eta,zeta) 0 0 funDN3Deta(ksi,eta,zeta) 0 0 funDN4Deta(ksi,eta,zeta) 0 ...
        0 funDN5Deta(ksi,eta,zeta) 0 0 funDN6Deta(ksi,eta,zeta) 0 0 funDN7Deta(ksi,eta,zeta) 0 0 funDN8Deta(ksi,eta,zeta) 0 ;
        0 funDN1Dzeta(ksi,eta,zeta) 0 0 funDN2Dzeta(ksi,eta,zeta) 0 0 funDN3Dzeta(ksi,eta,zeta) 0 0 funDN4Dzeta(ksi,eta,zeta) 0 ...
        0 funDN5Dzeta(ksi,eta,zeta) 0 0 funDN6Dzeta(ksi,eta,zeta) 0 0 funDN7Dzeta(ksi,eta,zeta) 0 0 funDN8Dzeta(ksi,eta,zeta) 0 ;
        0 0 funDN1Dksi(ksi,eta,zeta) 0 0 funDN2Dksi(ksi,eta,zeta) 0 0 funDN3Dksi(ksi,eta,zeta) 0 0 funDN4Dksi(ksi,eta,zeta) ...
        0 0 funDN5Dksi(ksi,eta,zeta) 0 0 funDN6Dksi(ksi,eta,zeta) 0 0 funDN7Dksi(ksi,eta,zeta) 0 0 funDN8Dksi(ksi,eta,zeta) ;
        0 0 funDN1Deta(ksi,eta,zeta) 0 0 funDN2Deta(ksi,eta,zeta) 0 0 funDN3Deta(ksi,eta,zeta) 0 0 funDN4Deta(ksi,eta,zeta) ...
        0 0 funDN5Deta(ksi,eta,zeta) 0 0 funDN6Deta(ksi,eta,zeta) 0 0 funDN7Deta(ksi,eta,zeta) 0 0 funDN8Deta(ksi,eta,zeta) ;
        0 0 funDN1Dzeta(ksi,eta,zeta) 0 0 funDN2Dzeta(ksi,eta,zeta) 0 0 funDN3Dzeta(ksi,eta,zeta) 0 0 funDN4Dzeta(ksi,eta,zeta) ...
        0 0 funDN5Dzeta(ksi,eta,zeta) 0 0 funDN6Dzeta(ksi,eta,zeta) 0 0 funDN7Dzeta(ksi,eta,zeta) 0 0 funDN8Dzeta(ksi,eta,zeta)];
    AregSum = AregSum + (alpha*(DN'))*DN;
end

% --- Element nodal DOF indices (identical construction to original) ---
tp = ones(1,DIM);
eleIndexU = zeros(NE,DIM*NodesPerEle);
for indEle = 1:NE
    tempIndexU = 3*elementsFEM(indEle,[tp,2*tp,3*tp,4*tp,5*tp,6*tp,7*tp,8*tp]);
    tempIndexU(1:3:end) = tempIndexU(1:3:end)-2;
    tempIndexU(2:3:end) = tempIndexU(2:3:end)-1;
    eleIndexU(indEle,:) = tempIndexU;
end

% --- Sparse triplet index lists (element-major, matching original order) ---
AI = zeros(576,NE); AJ = zeros(576,NE);
for indEle = 1:NE
    [IndexAXX,IndexAYY] = ndgrid(eleIndexU(indEle,:),eleIndexU(indEle,:));
    AI(:,indEle) = IndexAXX(:); AJ(:,indEle) = IndexAYY(:);
end
BIcol = reshape(eleIndexU.',[],1);

% --- Voxel offset grids within an element (integers, exact) ---
[offX,offY,offZ] = ndgrid(0:winsize(1),0:winsize(2),0:winsize(3));

% --- Chunking to bound peak memory of the batched interpolation ---
chunkSz = max(1, min(NE, floor(8e6/P)));

%% %%%%%%%%%%%%%%%% Start FE ICGN iterations %%%%%%%%%%%%%%%
for stepwithinwhile = 1:maxIter

    tic;
    if (stepwithinwhile==1)
        disp('--- Global IC-GN iterations (vectorized assembly) ---');
        AVAL = zeros(576,NE);
    end
    BVAL = zeros(24,NE);
    U = full(reshape(U,length(U),1));   % keep dense column (values unchanged)

    for e0 = 1:chunkSz:NE
        eChunk = e0:min(e0+chunkSz-1,NE);
        nC = numel(eChunk);
        QX = zeros(P,nC); QY = zeros(P,nC); QZ = zeros(P,nC);
        FV = zeros(P,nC); DFX = zeros(P,nC); DFY = zeros(P,nC); DFZ = zeros(P,nC);
        UeleC = zeros(24,nC);

        % ---- Gather subvolumes + build warped query points ----
        for ii = 1:nC
            indEle = eChunk(ii);
            p1 = pt1All(indEle,:);
            rx = p1(1):p1(1)+winsize(1); ry = p1(2):p1(2)+winsize(2); rz = p1(3):p1(3)+winsize(3);

            fsub = Img1(rx,ry,rz);                                       FV(:,ii)  = fsub(:);
            dsub = DfDx(rx-DfDxStartx,ry-DfDxStarty,rz-DfDxStartz);      DFX(:,ii) = dsub(:);
            dsub = DfDy(rx-DfDxStartx,ry-DfDxStarty,rz-DfDxStartz);      DFY(:,ii) = dsub(:);
            dsub = DfDz(rx-DfDxStartx,ry-DfDxStarty,rz-DfDxStartz);      DFZ(:,ii) = dsub(:);

            Uele = U(eleIndexU(indEle,:)); UeleC(:,ii) = Uele;
            % same accumulation order as original (bitwise-identical queries)
            tempUMat = zeros(winsize+ones(1,3)); tempVMat = tempUMat; tempWMat = tempUMat;
            for tempk = 1:NodesPerEle
                tempUMat = tempUMat + Uele(3*tempk-2)*NMat{tempk};
                tempVMat = tempVMat + Uele(3*tempk-1)*NMat{tempk};
                tempWMat = tempWMat + Uele(3*tempk-0)*NMat{tempk};
            end
            qx = (p1(1)+offX) + tempUMat;    % (p1+off) is exact integer = ndgrid pts
            qy = (p1(2)+offY) + tempVMat;
            qz = (p1(3)+offZ) + tempWMat;
            QX(:,ii) = qx(:); QY(:,ii) = qy(:); QZ(:,ii) = qz(:);
        end

        % ---- Batched deformed-volume sampling: g(x+u) ----
        TG = ba_interp3(Img2, QY, QX, QZ, 'cubic');   % same arg order as original

        % ---- Per-element assembly (matrix products replace voxel loop) ----
        for ii = 1:nC
            indEle = eChunk(ii);
            res = FV(:,ii) - TG(:,ii);                          % f - g(x+u), P x 1
            T3 = [DFX(:,ii).*res, DFY(:,ii).*res, DFZ(:,ii).*res]; % P x 3
            Mb = (NT8*T3).';                                    % 3 x 8 ; Mb(:) row 3(k-1)+c
            BVAL(:,indEle) = Mb(:) - AregSum*UeleC(:,ii);
            if (stepwithinwhile==1)
                Df24 = repmat([DFX(:,ii)'; DFY(:,ii)'; DFZ(:,ii)'],NodesPerEle,1); % 24 x P
                B = REPNT.*Df24;                                % B(3(k-1)+c,p) = N_k*Df_c
                tempA = B*B' + AregSum;
                AVAL(:,indEle) = tempA(:);
            end
        end
    end

    if (stepwithinwhile==1)
        A = sparse(AI(:),AJ(:),AVAL(:),FEMSize,FEMSize);
    end
    b = sparse(BIcol,ones(length(BIcol),1),BVAL(:),FEMSize,1);

    % ========= Solve FEM problem (identical to original) ===========
    W = A\b;

    normW = norm(W)/sqrt(size(W,1));
    normOfW(stepwithinwhile) = normW; %#ok<AGROW>
    timeICGN(stepwithinwhile) = toc; %#ok<AGROW>
    U = reshape(U,length(U),1); W = reshape(W,length(W),1);

    disp(['normW = ',num2str(normW),' at iter ',num2str(stepwithinwhile),'; time cost = ',num2str(toc),'s']);

    if stepwithinwhile == 1
        normWOld = normW*10;
    else
        normWOld = normOfW(stepwithinwhile-1);
    end

    if (normW < tol) || ((normW/normWOld > 0.9) && (normW/normWOld < 1))
        U = U + W;
        break;
    elseif (normW >= tol  && normW < (0.1/tol))
        U = U + W;
    else
        warning('Get diverged in Global_ICGN!!!')
        break;
    end

end

TotalTimeICGN = sum(timeICGN);
disp(['Elapsed time is ',num2str(TotalTimeICGN),' seconds.']);

end


%% ========= subroutines for FEM shape function derivatives (verbatim) ========
function a = funDN1Dksi(ksi,eta,zeta)
a = 1/8*(-1)*(1-eta)*(1-zeta);
end
function a = funDN2Dksi(ksi,eta,zeta)
a = 1/8*( 1)*(1-eta)*(1-zeta);
end
function a = funDN3Dksi(ksi,eta,zeta)
a = 1/8*( 1)*(1+eta)*(1-zeta);
end
function a = funDN4Dksi(ksi,eta,zeta)
a = 1/8*(-1)*(1+eta)*(1-zeta);
end
function a = funDN5Dksi(ksi,eta,zeta)
a = 1/8*(-1)*(1-eta)*(1+zeta);
end
function a = funDN6Dksi(ksi,eta,zeta)
a = 1/8*( 1)*(1-eta)*(1+zeta);
end
function a = funDN7Dksi(ksi,eta,zeta)
a = 1/8*( 1)*(1+eta)*(1+zeta);
end
function a = funDN8Dksi(ksi,eta,zeta)
a = 1/8*(-1)*(1+eta)*(1+zeta);
end

% ----------------------------------------------------
function a = funDN1Deta(ksi,eta,zeta)
a = 1/8*(1-ksi)*(-1)*(1-zeta);
end
function a = funDN2Deta(ksi,eta,zeta)
a = 1/8*(1+ksi)*(-1)*(1-zeta);
end
function a = funDN3Deta(ksi,eta,zeta)
a = 1/8*(1+ksi)*( 1)*(1-zeta);
end
function a = funDN4Deta(ksi,eta,zeta)
a = 1/8*(1-ksi)*( 1)*(1-zeta);
end
function a = funDN5Deta(ksi,eta,zeta)
a = 1/8*(1-ksi)*(-1)*(1+zeta);
end
function a = funDN6Deta(ksi,eta,zeta)
a = 1/8*(1+ksi)*(-1)*(1+zeta);
end
function a = funDN7Deta(ksi,eta,zeta)
a = 1/8*(1+ksi)*( 1)*(1+zeta);
end
function a = funDN8Deta(ksi,eta,zeta)
a = 1/8*(1-ksi)*( 1)*(1+zeta);
end

% ----------------------------------------------------
function a = funDN1Dzeta(ksi,eta,zeta)
a = 1/8*(1-ksi)*(1-eta)*(-1);
end
function a = funDN2Dzeta(ksi,eta,zeta)
a = 1/8*(1+ksi)*(1-eta)*(-1);
end
function a = funDN3Dzeta(ksi,eta,zeta)
a = 1/8*(1+ksi)*(1+eta)*(-1);
end
function a = funDN4Dzeta(ksi,eta,zeta)
a = 1/8*(1-ksi)*(1+eta)*(-1);
end
function a = funDN5Dzeta(ksi,eta,zeta)
a = 1/8*(1-ksi)*(1-eta)*( 1);
end
function a = funDN6Dzeta(ksi,eta,zeta)
a = 1/8*(1+ksi)*(1-eta)*( 1);
end
function a = funDN7Dzeta(ksi,eta,zeta)
a = 1/8*(1+ksi)*(1+eta)*( 1);
end
function a = funDN8Dzeta(ksi,eta,zeta)
a = 1/8*(1-ksi)*(1+eta)*( 1);
end
