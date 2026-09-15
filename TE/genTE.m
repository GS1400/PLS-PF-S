function genTE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% genTE.m
%
% Generates unconstrained monotone-equation test problems associated with
% three learning-based min--max models:
%
%   (1) Multi-Agent Reinforcement Learning (MARL)
%   (2) Robust Adversarial Learning
%   (3) Generative Adversarial Networks (GANs)
%
% Each problem is defined on R^n and stores a monotone operator E together
% with its convex-concave saddle objective. The corresponding saddle
% operator has the form
%
%       E(x) = [ grad_min Phi ; -grad_max Phi ].
%
% Each problem structure contains
%
%   P.funE       monotone operator E
%   P.funf       saddle objective Phi
%   P.resf       residual merit 0.5*||E(x)||^2
%   P.L          Lipschitz constant or valid global upper bound
%   P.x          initial point
%   P.xstar      known equilibrium/root
%   P.initLow    lower endpoint of the sampling range
%   P.initUpp    upper endpoint of the sampling range
%
% The finite sampling ranges are used to generate initial points and
% validation samples. The optimization problems themselves are
% unconstrained.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Store TE.mat in the same directory as genTE.m.
TEMEpath = fileparts(mfilename('fullpath'));

Nproblems = 100;
rng(42);

fprintf('\n=== Generating monotone-equation benchmarks ===\n\n');

% Remove a previously generated test-environment file.
tefile = fullfile(TEMEpath,'TE.mat');

if isfile(tefile)
    delete(tefile);
    fprintf('Deleted existing TE.mat: %s\n',tefile);
end

TE.problem = struct();

nameList = cell(1,Nproblems);
abbrList = cell(1,Nproblems);
dimList  = cell(1,Nproblems);

families = {'marl','robust_adversarial','gan'};

for i = 1:Nproblems

    family  = families{randi(3)};
    variant = randi(4);

    P = struct();

    switch family

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 1. Multi-Agent Reinforcement Learning
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Consider a one-state discounted two-player zero-sum Markov game.
        % For stationary mixed policies p and q,
        %
        %   Phi(p,q) = (1/(1-gamma)) p' Rstage q.
        %
        % With Rstage=(1-gamma)R,
        %
        %   Phi(p,q) = p'Rq.
        %
        % The reduced policy variables are
        %
        %   p = [u ; 1-sum(u)],
        %   q = [v ; 1-sum(v)],
        %
        % where u,v belong to R^(m-1). The resulting bilinear objective is
        % defined on R^(2m-2).
        %
        % The matrix R satisfies
        %
        %   R*1 = 0,    R'*1 = 0,
        %
        % so the uniform mixed policies define a known equilibrium. The
        % corresponding saddle operator has a skew-symmetric Jacobian and
        % is globally monotone and globally Lipschitz.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        case 'marl'

            switch variant
                case 1
                    m = randi([5,10]);
                    gamma = 0.90;
                    condLevel = 2;

                case 2
                    m = randi([8,15]);
                    gamma = 0.95;
                    condLevel = 5;

                case 3
                    m = randi([12,20]);
                    gamma = 0.98;
                    condLevel = 20;

                case 4
                    m = randi([18,30]);
                    gamma = 0.99;
                    condLevel = 100;
            end

            % Basis of the subspace orthogonal to the all-ones vector.
            Q = null(ones(1,m));

            % Construct the interaction matrix on the tangent subspace.
            [U0,~,V0] = svd(randn(m-1,m-1),'econ');

            sval = logspace(0,-log10(condLevel),m-1);

            R = Q*U0*diag(sval)*V0'*Q';

            % Enforce zero row and column sums numerically.
            J = eye(m) - ones(m)/m;
            R = J*R*J;

            % One-step reward matrix.
            Rstage = (1-gamma)*R;

            % Reduced policy map:
            %
            %   p = e_m + S*u,
            %
            % with
            %
            %       S = [ I
            %            -1' ].
            S = [eye(m-1); -ones(1,m-1)];
            eLast = [zeros(m-1,1);1];

            ustar = ones(m-1,1)/m;
            vstar = ones(m-1,1)/m;

            % Reduced bilinear coupling.
            C = S'*R*S;

            % Constant terms induced by the reduced policy map.
            cu = S'*R*eLast;
            cv = -S'*R'*eLast;

            % The Jacobian has the block form
            %
            %       M = [ 0    C
            %            -C'   0 ],
            %
            % hence ||C||_2 is the exact Lipschitz constant.
            L = norm(C,2);

            P.variant = sprintf('discounted_markov_v%d',variant);

            P.gamma    = gamma;
            P.R        = R;
            P.Rstage   = Rstage;
            P.S        = S;
            P.eLast    = eLast;
            P.nActions = m;

            P.dim   = 2*(m-1);
            P.xstar = [ustar;vstar];

            % Sampling range for initialization and validation.
            ub = 1/(m-1);

            P.initLow = zeros(P.dim,1);
            P.initUpp = ub*ones(P.dim,1);

            P.L = max(L,eps);

            % Self-contained function handles stored in TE.mat.
            rr = m-1;

            Efun = @(x) [ ...
                S'*(R*(eLast + S*x(rr+1:2*rr))); ...
               -S'*(R'*(eLast + S*x(1:rr))) ];

            Ffun = @(x) ...
                ((eLast + S*x(1:rr))' * ...
                 (Rstage*(eLast + S*x(rr+1:2*rr))))/(1-gamma);

            P.funE = Efun;
            P.funf = Ffun;
            P.resf = @(x) 0.5*sum(Efun(x).^2);

            % Select a nontrivial initial point.
            P.x = choose_initial_point( ...
                P.funE,P.initLow,P.initUpp,1e-2,1.0);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 2. Robust Adversarial Learning
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Consider
        %
        %   min_{theta in R^p} max_{delta in R^n} Phi(theta,delta),
        %
        % where e=theta-thetaStar and
        %
        %   Phi(theta,delta)
        %      = (1/(2n)) ||A e||^2
        %        + (1/n) e' B delta
        %        - (1/(2n)) sum_j eta_j delta_j^2.
        %
        % The quadratic term is concave in delta on R^n. The associated
        % saddle operator is
        %
        %   E = [ (A'A/n)e + (B/n)delta ;
        %         (eta/n).*delta - (B'/n)e ].
        %
        % Its symmetric Jacobian part is
        %
        %   diag(A'A/n,diag(eta/n)) >= 0,
        %
        % so E is globally monotone and globally Lipschitz.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        case 'robust_adversarial'

            switch variant
                case 1
                    p = randi([10,20]);
                    n = randi([30,50]);
                    deltaInit = 0.25;
                    etaBase = 0.50;
                    rankMode = 'full';

                case 2
                    p = randi([25,40]);
                    n = randi([15,24]);
                    deltaInit = 0.50;
                    etaBase = 0.20;
                    rankMode = 'deficient';

                case 3
                    p = randi([40,60]);
                    n = randi([20,35]);
                    deltaInit = 0.75;
                    etaBase = 0.05;
                    rankMode = 'deficient';

                case 4
                    p = randi([60,90]);
                    n = randi([25,45]);
                    deltaInit = 1.00;
                    etaBase = 0.01;
                    rankMode = 'deficient';
            end

            if strcmp(rankMode,'full')

                A = randn(n,p)/sqrt(p);

                while rank(A) < p
                    A = randn(n,p)/sqrt(p);
                end

            else

                A = randn(n,p)/sqrt(p);

            end

            B = randn(p,n)/sqrt(p);

            eta = etaBase*(0.75 + 0.5*rand(n,1));

            thetaStar = 0.2*randn(p,1);

            H  = (A'*A)/n;
            Bn = B/n;
            D  = diag(eta/n);

            M = [H, Bn; -Bn', D];

            P.variant = sprintf('adversarial_v%d',variant);

            P.A         = A;
            P.B         = B;
            P.eta       = eta;
            P.deltaInit = deltaInit;
            P.thetaStar = thetaStar;
            P.nTheta    = p;
            P.nSamples  = n;

            P.dim   = p+n;
            P.xstar = [thetaStar;zeros(n,1)];

            % Sampling range for initialization and validation.
            thetaBound = max(2,2*max(abs(thetaStar))+1);

            P.initLow = [-thetaBound*ones(p,1); ...
                         -deltaInit*ones(n,1)];

            P.initUpp = [ thetaBound*ones(p,1); ...
                          deltaInit*ones(n,1)];

            P.L = max(norm(M,2),eps);

            % Self-contained function handles stored in TE.mat.
            pp = p;
            nn = n;

            Efun = @(x) [ ...
                (A'*(A*(x(1:pp)-thetaStar)) + ...
                 B*x(pp+1:pp+nn))/nn; ...
                (eta.*x(pp+1:pp+nn) - ...
                 B'*(x(1:pp)-thetaStar))/nn ];

            Ffun = @(x) ...
                0.5*sum((A*(x(1:pp)-thetaStar)).^2)/nn + ...
                ((x(1:pp)-thetaStar)' * ...
                 (B*x(pp+1:pp+nn)))/nn - ...
                0.5*sum(eta.*x(pp+1:pp+nn).^2)/nn;

            P.funE = Efun;
            P.funf = Ffun;
            P.resf = @(x) 0.5*sum(Efun(x).^2);

            % Select a nontrivial initial point.
            P.x = choose_initial_point( ...
                P.funE,P.initLow,P.initUpp,1e-2,1.0);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 3. Generative Adversarial Networks
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Consider the globally convex-concave categorical/logistic
        % surrogate
        %
        %   Phi(u,d)
        %      = 0.5*(pdata-pG(u))'d
        %        - sum_j log(2*cosh(d_j/2)),
        %
        % where
        %
        %   pG(u) = eLast+S*u,
        %   u in R^(m-1),
        %   d in R^m.
        %
        % Since
        %
        %   log(2*cosh(d/2)) = softplus(d)-d/2,
        %
        % the objective can be evaluated stably. The corresponding saddle
        % operator is
        %
        %   E_u = -0.5*S'*d,
        %
        %   E_d = 0.5*(pG(u)-pdata)+sigmoid(d)-0.5.
        %
        % The symmetric part of the Jacobian is positive semidefinite, so
        % E is globally monotone. A global Lipschitz upper bound is
        %
        %   L <= 0.5*sqrt(m)+0.25.
        %
        % A known root is given by pG=pdata and d=0.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        case 'gan'

            switch variant
                case 1
                    m = randi([5,10]);
                    dInit = 6;
                    spread = 0.02;

                case 2
                    m = randi([8,15]);
                    dInit = 8;
                    spread = 0.015;

                case 3
                    m = randi([12,20]);
                    dInit = 10;
                    spread = 0.010;

                case 4
                    m = randi([18,30]);
                    dInit = 12;
                    spread = 0.005;
            end

            % Construct a strictly positive categorical reference
            % distribution.
            ub   = 1/(m-1);
            pmin = 0.25/m;
            pmax = 0.99*ub;

            base = ones(m,1)/m;

            z = randn(m,1);
            z = z - mean(z);

            % Set the perturbation magnitude.
            z = spread*z/max(1,norm(z,inf));

            % Determine a scaling that preserves the sampling margins.
            alpha = 1.0;

            for jj = 1:m

                if z(jj) < 0
                    alpha = min(alpha, ...
                        (base(jj)-pmin)/(-z(jj)));
                end

                if jj <= m-1 && z(jj) > 0
                    alpha = min(alpha, ...
                        (pmax-base(jj))/z(jj));
                end

            end

            alpha = max(0,min(1,0.95*alpha));

            pdata = base + alpha*z;

            % Normalize the reference distribution.
            pdata = max(pdata,pmin);
            pdata = pdata/sum(pdata);

            % Verify positivity and the reduced-coordinate range.
            if any(pdata <= 0) || ...
               any(pdata(1:m-1) >= ub) || ...
               abs(sum(pdata)-1) > 1e-12

                error('Failed to construct a valid pdata reference.');

            end

            S = [eye(m-1); -ones(1,m-1)];
            eLast = [zeros(m-1,1);1];

            ustar = pdata(1:m-1);
            dstar = zeros(m,1);

            P.variant = sprintf( ...
                'categorical_logistic_gan_v%d',variant);

            P.pdata = pdata;
            P.S     = S;
            P.eLast = eLast;
            P.nData = m;
            P.dInit = dInit;

            P.dim   = (m-1)+m;
            P.xstar = [ustar;dstar];

            % Sampling range for initialization and validation.
            P.initLow = [zeros(m-1,1); ...
                         -dInit*ones(m,1)];

            P.initUpp = [ub*ones(m-1,1); ...
                          dInit*ones(m,1)];

            % Global Lipschitz upper bound.
            P.L = 0.5*sqrt(m) + 0.25;

            % Self-contained function handles stored in TE.mat.
            rr = m-1;

            sigmoid  = @(z) 0.5*(1+tanh(z/2));
            softplus = @(z) max(z,0) + log1p(exp(-abs(z)));

            Efun = @(x) [ ...
                -0.5*S'*x(rr+1:rr+m); ...
                 0.5*((eLast + S*x(1:rr))-pdata) + ...
                 sigmoid(x(rr+1:rr+m)) - 0.5 ];

            Ffun = @(x) ...
                0.5*(pdata-(eLast + S*x(1:rr)))' * ...
                    x(rr+1:rr+m) - ...
                sum(softplus(x(rr+1:rr+m)) - ...
                    0.5*x(rr+1:rr+m));

            P.funE = Efun;
            P.funf = Ffun;
            P.resf = @(x) 0.5*sum(Efun(x).^2);

            % Select a nontrivial initial point.
            P.x = choose_initial_point( ...
                P.funE,P.initLow,P.initUpp,1e-2,1.0);

    end

    % Unbounded-domain fields used by the solver interface.
    P.low = -inf(P.dim,1);
    P.upp =  inf(P.dim,1);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Problem validation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    validate_problem(P,i,family);


    % Test monotonicity and the Lipschitz bound on sampled points.
    for j = 1:50

        xa = P.initLow + rand(P.dim,1).*(P.initUpp-P.initLow);
        xb = P.initLow + rand(P.dim,1).*(P.initUpp-P.initLow);

        Ea = P.funE(xa);
        Eb = P.funE(xb);

        if any(~isfinite(Ea)) || any(~isfinite(Eb))
            error('Problem %d produced NaN/Inf in funE.',i);
        end

        % Numerical monotonicity test.
        mon = (Ea-Eb)'*(xa-xb);

        monTol = 1e-10*(1 + norm(Ea-Eb)*norm(xa-xb));

        if mon < -monTol

            error(['Problem %d (%s) failed monotonicity check: ', ...
                   '<dE,dx> = %e.'],i,family,mon);

        end

        % Numerical Lipschitz test.
        dx = norm(xa-xb);

        if dx > 0

            ratio = norm(Ea-Eb)/dx;

            if ratio > P.L*(1+1e-8)

                error(['Problem %d (%s) failed Lipschitz check: ', ...
                       'ratio=%e > L=%e.'],i,family,ratio,P.L);

            end

        end

    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Store the problem
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    pname = sprintf('ME%d_%s_%s',i,family,P.variant);

    P.name = pname;

    TE.problem.(pname) = P;

    nameList{i} = sprintf('ME%d',i);
    abbrList{i} = pname;
    dimList{i}  = num2str(P.dim);

    fprintf(['Problem %3d: %-20s %-23s dim=%4d  ', ...
             'L=%9.3e  ||E(x*)||=%8.1e\n'], ...
             i,family,P.variant,P.dim,P.L,norm(P.funE(P.xstar)));

    fprintf('             initial residual ||E(x0)|| = %.6e\n', ...
            norm(P.funE(P.x)));

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save TE.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TE.name     = nameList;
TE.nameAbbr = abbrList;
TE.dim      = dimList;

if ~exist(TEMEpath,'dir')
    mkdir(TEMEpath);
end

save(tefile,'TE','-v7.3');

fprintf('Saved TE.mat to: %s\n',tefile);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verify the saved function handles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear TE

Scheck = load(tefile,'TE');
TEcheck = Scheck.TE;

checkNames = fieldnames(TEcheck.problem);

for kk = 1:numel(checkNames)

    Pc = TEcheck.problem.(checkNames{kk});

    Ec = Pc.funE(Pc.x);
    fc = Pc.funf(Pc.x);

    if numel(Ec) ~= Pc.dim || ...
       any(~isfinite(Ec(:))) || ...
       ~isfinite(fc)

        error('Saved-handle reload test failed for %s.', ...
              checkNames{kk});

    end

end

TE = TEcheck;

clear TEcheck Scheck

fprintf('\n=== TE generation complete ===\n');
fprintf('=== Saved function handles verified ===\n');

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose an initial point from the prescribed sampling range
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x0 = choose_initial_point(funE,initLow,initUpp,rmin,rmax)

n = length(initLow);

bestx = [];
bestr = -inf;

for trial = 1:2000

    xtry = initLow + rand(n,1).*(initUpp-initLow);
    Etry = funE(xtry);

    if any(~isfinite(Etry(:)))
        continue;
    end

    r = norm(Etry);

    if isfinite(r) && r >= rmin && r <= rmax

        x0 = xtry;
        return;

    end

    % Retain the finite sampled point with the largest residual.
    if isfinite(r) && r > bestr

        bestr = r;
        bestx = xtry;

    end

end

if isempty(bestx)
    error('Unable to construct a finite initial point.');
end

x0 = bestx;

fprintf(['WARNING: desired initial residual interval [%e,%e] ', ...
         'was not attained; using residual %e.\n'], ...
         rmin,rmax,bestr);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Common problem validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function validate_problem(P,idx,family)

if ~(isscalar(P.L) && isfinite(P.L) && P.L > 0)
    error('Problem %d has invalid P.L.',idx);
end

if numel(P.initLow) ~= P.dim || numel(P.initUpp) ~= P.dim
    error('Problem %d has invalid initialization-range dimensions.',idx);
end

if any(P.initLow >= P.initUpp)
    error('Problem %d has invalid initialization range.',idx);
end

if numel(P.x) ~= P.dim || numel(P.xstar) ~= P.dim
    error('Problem %d has inconsistent dimensions.',idx);
end

if any(P.x < P.initLow) || any(P.x > P.initUpp)
    error('Problem %d has initial point outside its sampling range.',idx);
end

E0 = P.funE(P.x);
Es = P.funE(P.xstar);

r0 = norm(E0);

if r0 < 1e-2
    warning('Problem %d starts too close to a root: ||E(x0)||=%e.', ...
            idx,r0);
end

if numel(E0) ~= P.dim || numel(Es) ~= P.dim
    error('Problem %d has incorrect funE dimension.',idx);
end

if any(~isfinite(E0)) || any(~isfinite(Es))
    error('Problem %d has NaN/Inf in funE.',idx);
end

if norm(Es) > 1e-9

    error('Problem %d (%s): xstar is not a root; residual=%e.', ...
          idx,family,norm(Es));

end

Phi0 = P.funf(P.x);
Phis = P.funf(P.xstar);
r0   = P.resf(P.x);

if ~isscalar(Phi0) || ~isfinite(Phi0)
    error('Problem %d has invalid funf(x0).',idx);
end

if ~isscalar(Phis) || ~isfinite(Phis)
    error('Problem %d has invalid funf(xstar).',idx);
end

if ~isscalar(r0) || ~isfinite(r0)
    error('Problem %d has invalid residual merit.',idx);
end

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MARL objective
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Phi = marl_value(x,Rstage,gamma,S,eLast)

r = size(S,2);

u = x(1:r);
v = x(r+1:2*r);

p = eLast + S*u;
q = eLast + S*v;

Phi = (p'*(Rstage*q))/(1-gamma);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MARL monotone operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = marl_operator(x,R,S,eLast)

r = size(S,2);

u = x(1:r);
v = x(r+1:2*r);

p = eLast + S*u;
q = eLast + S*v;

Eu =  S'*(R*q);
Ev = -S'*(R'*p);

E = [Eu;Ev];

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust adversarial objective
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Phi = adversarial_value(x,A,B,eta,thetaStar)

p = length(thetaStar);
n = length(eta);

theta = x(1:p);
delta = x(p+1:p+n);

e = theta-thetaStar;

Phi = 0.5*sum((A*e).^2)/n ...
      + (e'*(B*delta))/n ...
      - 0.5*sum(eta.*delta.^2)/n;

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust adversarial monotone operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = adversarial_operator(x,A,B,eta,thetaStar)

p = length(thetaStar);
n = length(eta);

theta = x(1:p);
delta = x(p+1:p+n);

e = theta-thetaStar;

Etheta = (A'*(A*e) + B*delta)/n;
Edelta = (eta.*delta - B'*e)/n;

E = [Etheta;Edelta];

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Categorical/logistic GAN objective
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Phi = gan_value(x,pdata,S,eLast)

r = size(S,2);
m = length(pdata);

u = x(1:r);
d = x(r+1:r+m);

pG = eLast + S*u;

Phi = 0.5*(pdata-pG)'*d ...
      - sum(stable_softplus(d)-0.5*d);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Categorical/logistic GAN monotone operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = gan_operator(x,pdata,S,eLast)

r = size(S,2);
m = length(pdata);

u = x(1:r);
d = x(r+1:r+m);

pG = eLast + S*u;

Eu = -0.5*S'*d;
Ed = 0.5*(pG-pdata) + stable_sigmoid(d) - 0.5;

E = [Eu;Ed];

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stable sigmoid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = stable_sigmoid(z)

y = 0.5*(1+tanh(z/2));

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stable softplus
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = stable_softplus(z)

y = max(z,0) + log1p(exp(-abs(z)));

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Residual merit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = residual_merit(E)

if any(~isfinite(E(:)))

    f = realmax;
    return;

end

r = norm(E);

if ~isfinite(r) || r > sqrt(realmax)

    f = realmax;

else

    f = 0.5*r*r;

end

end
