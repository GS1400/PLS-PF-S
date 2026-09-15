function driverME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% driverME.m
%
% The stopping mechanism is implemented inside the operator evaluation,
% analogously to getfLS.
%
% Existing solver files called directly:
%
%   DFSdir.m
%   MonotoneSolvers.m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

driverDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(driverDir);

solverPath = ...
fullfile(repoRoot,'solvers');

TEpath = ...
fullfile(repoRoot,'TE');

TEfile = ...
fullfile(TEpath,'TE.mat');

figDir = ...
fullfile(repoRoot,'results','figures');


if exist(solverPath,'dir')~=7
    error('Solver directory not found:\n%s',solverPath);
end

if exist(TEfile,'file')~=2
    error('TE.mat not found:\n%s',TEfile);
end

if exist(figDir,'dir')~=7
    mkdir(figDir);
end


addpath(solverPath,'-begin');


fprintf('\nSolver files used:\n');
fprintf('DFSdir          : %s\n',which('DFSdir'));
fprintf('MonotoneSolvers : %s\n\n',which('MonotoneSolvers'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load TE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = load(TEfile,'TE');

TE = S.TE;

problemFields = fieldnames(TE.problem);

for k = 1:numel(problemFields)

    pname = problemFields{k};

    TE.problem.(pname) = ...
        attachProblemFunctions(TE.problem.(pname),pname);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The current TE.mat has been regenerated.  The three problems below are
% chosen as structural counterparts of the previously reported examples:
%
%   MARL : variant 4, dim = 46
%   RAL  : variant 1, dim = 55
%   GAN  : variant 4, dim = 53
%
% In particular, the updated GAN family is named
% categorical_logistic_gan rather than standard_log_gan.

marlName = ...
'ME54_marl_discounted_markov_v4';

advName = ...
'ME71_robust_adversarial_adversarial_v1';

ganName = ...
'ME65_gan_categorical_logistic_gan_v4';


problemNames = { ...
    marlName, ...
    advName, ...
    ganName};


problemTitles = { ...
    'MARL', ...
    'Robust adversarial learning', ...
    'GAN'};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

solverNames = { ...
    'PLS-S-z2', ...
    'EG', ...
    'OGDA'};


nSolver = numel(solverNames);

nProblem = numel(problemNames);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Benchmark stopping tests
%
% These are applied inside evalOperator, exactly where E is evaluated.
%
% The benchmark stopping quantity is
%
%       f(x) = 0.5 ||E(x)||^2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accf = 1e-5;

nfmax = 20000;

secmax = 300;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solver-side stopping settings
%
% Accuracy and time stopping are handled exclusively by evalOperator.
%
% The benchmark wrapper also supplies the residual-evaluation budget to
% DFSdir. We retain it here for consistency, although evalOperator enforces
% the same nfmax before an additional evaluation is permitted.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

st = struct();

st.prt = -1;

st.accf = -inf;

st.nfmax = nfmax;

st.secmax = inf;

st.fbest = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot styles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lineStyles = { ...
    '-', ...
    '--', ...
    '-.'};


markers = { ...
    '^', ...
    'o', ...
    's'};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Storage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RESULT = struct();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables shared with nested evaluation routine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = [];

phiStar = [];

evalInfo = struct();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main problem loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ip = 1:nProblem


    pname = problemNames{ip};

    P = TE.problem.(pname);


    x0 = P.x(:);


    phiStar = P.funf(P.xstar);


    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('Problem %d/%d: %s\n',ip,nProblem,pname);
    fprintf('Application: %s\n',problemTitles{ip});
    fprintf('dimension  : %d\n',P.dim);
    fprintf('L          : %.6e\n',P.L);
    fprintf('mu(PLS)    : %.16e\n',0.48/P.L);
    fprintf('||E(x0)||  : %.6e\n',norm(P.funE(x0)));
    fprintf('f(x0)      : %.6e\n',0.5*norm(P.funE(x0))^2);
    fprintf('============================================================\n');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Solver loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for is = 1:nSolver


        solver = solverNames{is};


        fprintf('\nRunning %-12s on %s ...\n',solver,pname);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Initialize evaluation information
        %
        % This plays the role of fginfo in getfLS.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        evalInfo = struct();


        evalInfo.nf = 0;

        evalInfo.nfmax = nfmax;

        evalInfo.secmax = secmax;

        evalInfo.accf = accf;

        evalInfo.cpu0 = cputime;


        evalInfo.fbest = inf;

        evalInfo.xbest = x0;

        evalInfo.nfbest = 0;


        evalInfo.error = '';


        evalInfo.hist_nf = [];

        evalInfo.hist_f = [];

        evalInfo.hist_res = [];

        evalInfo.hist_objerr = [];

        evalInfo.hist_dist = [];


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function handle supplied to solver
        %
        % Every operator evaluation passes through evalOperator.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        funEval = @evalOperator;


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Call original solver
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        solverInfo = struct();


        try


            switch solver


                %==========================================================
                % PF-S-LS
                %
                % IMPORTANT:
                %
                % Reproduce wPLS exactly:
                %
                %   solverParams{5} = struct('state','LS')
                %   tune.mu = 0.48/options.L
                %   tune.L  = options.L
                %
                % All other PLS-S parameters are intentionally left
                % unspecified so DFSdir uses the same defaults as in the
                % benchmark run.
                %==========================================================

                case 'PLS-S-z2'


                    tune = struct();

                    tune.state = '3HS';

                    tune.mu = 0.48/P.L;

                    tune.L = P.L;


                    [xfinal,ffinal,solverInfo] = ...
                        DFSdir( ...
                        funEval, ...
                        x0, ...
                        st, ...
                        tune);


                %==========================================================
                % EG
                %==========================================================

                case 'EG'


                    tune = struct();

                    tune.method = 'EG';

                    tune.L = P.L;

                    tune.maxit = inf;


                    [xfinal,ffinal,solverInfo] = ...
                        MonotoneSolvers( ...
                        funEval, ...
                        x0, ...
                        st, ...
                        tune);


                %==========================================================
                % OGDA
                %==========================================================

                case 'OGDA'


                    tune = struct();

                    tune.method = 'OGDA';

                    tune.L = P.L;

                    tune.maxit = inf;


                    [xfinal,ffinal,solverInfo] = ...
                        MonotoneSolvers( ...
                        funEval, ...
                        x0, ...
                        st, ...
                        tune);


                otherwise


                    error('Unknown solver %s.',solver);


            end


        catch ME


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Expected stopping exception from evalOperator
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if strcmp(ME.identifier,'LearningTest:Stop')


                xfinal = evalInfo.xbest;

                ffinal = evalInfo.fbest;


            else


                rethrow(ME);


            end


        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % If solver returned normally, use the best point seen by evaluator
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if evalInfo.fbest < inf

            xfinal = evalInfo.xbest;

            ffinal = evalInfo.fbest;

        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Build information structure
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        info = struct();


        info.nf = evalInfo.nf;

        info.nfmax = evalInfo.nfmax;

        info.secmax = evalInfo.secmax;

        info.accf = evalInfo.accf;


        info.fbest = evalInfo.fbest;

        info.xbest = evalInfo.xbest;

        info.nfbest = evalInfo.nfbest;


        info.error = evalInfo.error;

        info.sec = cputime-evalInfo.cpu0;


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Save histories
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        fieldSolver = ...
            matlab.lang.makeValidName(solver);


        RESULT.(pname).(fieldSolver).nf = ...
            evalInfo.hist_nf;


        RESULT.(pname).(fieldSolver).f = ...
            evalInfo.hist_f;


        RESULT.(pname).(fieldSolver).res = ...
            evalInfo.hist_res;


        RESULT.(pname).(fieldSolver).objerr = ...
            evalInfo.hist_objerr;


        RESULT.(pname).(fieldSolver).dist = ...
            evalInfo.hist_dist;


        RESULT.(pname).(fieldSolver).x = ...
            xfinal;


        RESULT.(pname).(fieldSolver).ffinal = ...
            ffinal;


        RESULT.(pname).(fieldSolver).info = ...
            info;


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Final information
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Efinal = P.funE(xfinal);

        phiFinal = P.funf(xfinal);


        fprintf( ...
            ['Finished %-12s: nf=%6d, ', ...
             'fbest=%10.4e, ', ...
             '||E||=%10.4e, ', ...
             '|Phi-Phi*|=%10.4e, ', ...
             '||x-x*||=%10.4e, ', ...
             '%s\n'], ...
             solver, ...
             evalInfo.nf, ...
             evalInfo.fbest, ...
             norm(Efinal), ...
             abs(phiFinal-phiStar), ...
             norm(xfinal-P.xstar(:)), ...
             evalInfo.error);


        fclose('all');


    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Merit-function figure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    fig0 = figure( ...
        'Name',[problemTitles{ip},' merit function'], ...
        'Color','w');


    hold on


    for is = 1:nSolver


        solver = solverNames{is};

        fieldSolver = matlab.lang.makeValidName(solver);


        nf = RESULT.(pname).(fieldSolver).nf;

        fvals = RESULT.(pname).(fieldSolver).f;


        if isempty(nf)
            continue
        end


        markerStep = max(1,floor(numel(nf)/12));

        markerInd = 1:markerStep:numel(nf);


        plot( ...
            nf, ...
            log10(max(fvals,realmin)), ...
            'LineWidth',1.8, ...
            'LineStyle',lineStyles{is}, ...
            'Marker',markers{is}, ...
            'MarkerIndices',markerInd, ...
            'DisplayName',solver);


    end


    yline( ...
        log10(accf), ...
        '--', ...
        'Tolerance', ...
        'FontWeight','bold', ...
        'HandleVisibility','off');


    hold off


%     xlabel( ...
%         'Function evaluations, nf', ...
%         'FontWeight','bold');
% 
% 
%     ylabel( ...
%         '$\log_{10}\!\left(\frac{1}{2}\|E(x)\|^2\right)$', ...
%         'Interpreter','latex', ...
%         'FontWeight','bold');
% 
% 
%     title( ...
%         sprintf('%s: merit-function convergence',problemTitles{ip}), ...
%         'FontWeight','bold');


    grid on
    box on


    lgd = legend( ...
        'Location','best', ...
        'Interpreter','none');

    set(lgd,'FontWeight','bold');


    set( ...
        gca, ...
        'FontSize',11, ...
        'FontWeight','bold');


    figName = ...
        fullfile(figDir,[pname,'_log_merit_nf']);


    exportgraphics( ...
        fig0, ...
        [figName,'.png'], ...
        'Resolution',300);


    print( ...
        fig0, ...
        [figName,'.eps'], ...
        '-depsc', ...
        '-painters');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Min--max objective-error figure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    fig2 = figure( ...
        'Name',[problemTitles{ip},' objective error'], ...
        'Color','w');


    hold on


    for is = 1:nSolver


        solver = solverNames{is};

        fieldSolver = matlab.lang.makeValidName(solver);


        nf = RESULT.(pname).(fieldSolver).nf;

        objerr = RESULT.(pname).(fieldSolver).objerr;


        if isempty(nf)
            continue
        end


        markerStep = max(1,floor(numel(nf)/12));

        markerInd = 1:markerStep:numel(nf);


        plot( ...
            nf, ...
            log10(max(objerr,realmin)), ...
            'LineWidth',1.8, ...
            'LineStyle',lineStyles{is}, ...
            'Marker',markers{is}, ...
            'MarkerIndices',markerInd, ...
            'DisplayName',solver);


    end


    hold off


%     xlabel( ...
%         'Function evaluations, nf', ...
%         'FontWeight','bold');
% 
% 
%     ylabel( ...
%         '$\log_{10}|\Phi(x)-\Phi(x^*)|$', ...
%         'Interpreter','latex', ...
%         'FontWeight','bold');
% 
% 
%     title( ...
%         sprintf('%s: min--max objective error',problemTitles{ip}), ...
%         'Interpreter','latex', ...
%         'FontWeight','bold');


    grid on
    box on


    lgd = legend( ...
        'Location','best', ...
        'Interpreter','none');

    set(lgd,'FontWeight','bold');


    set( ...
        gca, ...
        'FontSize',11, ...
        'FontWeight','bold');


    figName = ...
        fullfile(figDir,[pname,'_log_objective_error_nf']);


    exportgraphics( ...
        fig2, ...
        [figName,'.png'], ...
        'Resolution',300);


    print( ...
        fig2, ...
        [figName,'.eps'], ...
        '-depsc', ...
        '-painters');


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save histories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

resultFile = ...
fullfile(figDir,'three_learning_examples_results.mat');


save( ...
    resultFile, ...
    'RESULT', ...
    'problemNames', ...
    'problemTitles', ...
    'solverNames', ...
    'accf', ...
    'nfmax', ...
    'secmax');


fprintf('\n');
fprintf('============================================================\n');
fprintf('All runs completed.\n');
fprintf('Results saved in:\n%s\n',resultFile);
fprintf('============================================================\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation routine
%
% Analogue of getfLS.
%
% Every request for E(x) enters here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [E,f] = evalOperator(xcur)


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stopping tests BEFORE the next evaluation
    %
    % This reproduces the getfLS logic:
    %
    %   accuracy first,
    %   then nfmax,
    %   then secmax.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    evalInfo.error = '';


    if evalInfo.fbest <= evalInfo.accf


        evalInfo.error = 'accuracy reached';


    elseif evalInfo.nf >= evalInfo.nfmax


        evalInfo.error = 'nfmax reached';


    elseif cputime-evalInfo.cpu0 >= evalInfo.secmax


        evalInfo.error = 'secmax reached';


    end


    if ~isempty(evalInfo.error)


        fprintf('\n');
        fprintf('============================================================\n');
        fprintf('%s\n',evalInfo.error);
        fprintf('nf    = %d\n',evalInfo.nf);
        fprintf('fbest = %.16e\n',evalInfo.fbest);
        fprintf('============================================================\n');


        error( ...
            'LearningTest:Stop', ...
            evalInfo.error);


    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Evaluate operator
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    xcur = xcur(:);


    E = P.funE(xcur);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Same merit function as getfLS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if any(~isfinite(E(:)))


        f = 1e50;


    else


        r = norm(E);


        if ~isfinite(r) || r >= sqrt(2e50)


            f = 1e50;


        else


            f = 0.5*r*r;


        end


    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Count evaluation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    evalInfo.nf = ...
        evalInfo.nf+1;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Store histories
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    evalInfo.hist_nf(end+1,1) = ...
        evalInfo.nf;


    evalInfo.hist_f(end+1,1) = ...
        f;


    evalInfo.hist_res(end+1,1) = ...
        norm(E);


    phi = ...
        P.funf(xcur);


    evalInfo.hist_objerr(end+1,1) = ...
        abs(phi-phiStar);


    evalInfo.hist_dist(end+1,1) = ...
        norm(xcur-P.xstar(:));


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update best merit point
    %
    % Same principle as getfLS:
    %
    %       if f <= fbest
    %
    % update the best point and evaluation number.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if f <= evalInfo.fbest


        evalInfo.fbest = ...
            f;


        evalInfo.xbest = ...
            xcur;


        evalInfo.nfbest = ...
            evalInfo.nf;


    end


end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruct problem functions from the stored numerical data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = attachProblemFunctions(P,pname)


if contains(pname,'_marl_')

    R      = P.R;
    Rstage = P.Rstage;
    S      = P.S;
    eLast  = P.eLast;
    gamma  = P.gamma;

    rr = size(S,2);

    Efun = @(x) [ ...
        S'*(R*(eLast + S*x(rr+1:2*rr))); ...
       -S'*(R'*(eLast + S*x(1:rr))) ];

    Ffun = @(x) ...
        ((eLast + S*x(1:rr))' * ...
         (Rstage*(eLast + S*x(rr+1:2*rr))))/(1-gamma);


elseif contains(pname,'_robust_adversarial_')

    A         = P.A;
    B         = P.B;
    eta       = P.eta;
    thetaStar = P.thetaStar;

    pp = P.nTheta;
    nn = P.nSamples;

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


elseif contains(pname,'_gan_')

    pdata  = P.pdata;
    S      = P.S;
    eLast  = P.eLast;

    rr = size(S,2);
    mm = P.nData;

    sigmoid  = @(z) 0.5*(1+tanh(z/2));
    softplus = @(z) max(z,0) + log1p(exp(-abs(z)));

    Efun = @(x) [ ...
        -0.5*S'*x(rr+1:rr+mm); ...
         0.5*((eLast + S*x(1:rr))-pdata) + ...
         sigmoid(x(rr+1:rr+mm)) - 0.5 ];

    Ffun = @(x) ...
        0.5*(pdata-(eLast + S*x(1:rr)))' * ...
            x(rr+1:rr+mm) - ...
        sum(softplus(x(rr+1:rr+mm)) - ...
            0.5*x(rr+1:rr+mm));


else

    error('Unknown problem family: %s',pname);

end


P.funE = Efun;
P.funf = Ffun;
P.resf = @(x) 0.5*sum(Efun(x).^2);


end