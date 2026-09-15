function [x,f,info] = MonotoneSolvers(fun,x0,st,tune)
% MONOTONESOLVERS  Comparison methods for unconstrained monotone equations.
%
% The routine applies first-order methods to E(x)=0 with x in R^n.
%
% Methods:
%   'EG'     Extragradient
%   'OGDA'   Optimistic Gradient Descent-Ascent
%   'EAG'    Extra Anchored Gradient
%   'FOGDA'  Fast OGDA
%
% INPUT:
%   fun    : function handle E(x)
%   x0     : initial point in R^n
%   tune.method   : 'EG','OGDA','EAG','FOGDA'
%   tune.s        : stepsize
%   tune.L        : optional Lipschitz constant
%   tune.maxit    : max iterations
%   tune.tol      : stopping tolerance
%
% OUTPUT:
%   x      : final iterate
%   f      : 0.5*||E(x)||^2
%   info   : struct with residual history
%
% All methods are implemented in unconstrained form. No box projection,
% clipping, or application-specific feasibility correction is applied.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Default parameters.
if ~isfield(tune,'method'), tune.method = 'EG'; end
if ~isfield(tune,'s'),      tune.s = 1e-2; end
if ~isfield(tune,'maxit'),  tune.maxit = inf; end
if ~isfield(tune,'tol'),    tune.tol = 1e-6; end

method = tune.method;
s      = tune.s;

% Diagnostic output.
if isfield(tune,'debug'), debug = tune.debug;
else, debug = 0;
end

if isfield(tune,'debugEvery'), debugEvery = tune.debugEvery;
else, debugEvery = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unconstrained initial point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x0 = x0(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optional step-size safeguard using the Lipschitz constant.
%
% If tune.L is supplied, the prescribed step is reduced only when it
% exceeds the method-specific bound. Otherwise tune.s is used unchanged.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isfield(tune,'L') && ~isempty(tune.L)

    L = tune.L;

    if isfinite(L) && L > 0

        switch upper(method)

            case 'EG'
                smax = 0.99/L;

            case 'OGDA'
                smax = 0.49/L;

            case 'EAG'
                smax = 0.25/L;

            case 'FOGDA'
                smax = 0.10/L;

            otherwise
                smax = inf;
        end

        if s > smax
            s = smax;
        end
    end
end

if debug
    fprintf('\n================ MONOTONE SOLVER DEBUG ================\n');
    fprintf('method = %s\n',method);
    fprintf('stepsize s = %.16e\n',s);
    if isfield(tune,'L') && ~isempty(tune.L)
        fprintf('L = %.16e\n',tune.L);
        fprintf('s*L = %.16e\n',s*tune.L);
    else
        fprintf('L = <not supplied>\n');
    end
    fprintf('=========================================================\n\n');
end


% Performance and stopping information
%  .prt          printing level
%  .secmax       maximum CPU time
%  .nfmax        maximum number of operator evaluations
%  .fbest        target merit value
%  .accf         residual stopping tolerance
%  .qso          current residual norm
%  .initTime     initial CPU time
%  .done         stopping flag
%
% Printing level.
if isfield(st,'prt'), info.prt = st.prt;
else info.prt = -1;
end

% Stopping criteria.
if isfield(st,'secmax'), info.secmax=st.secmax;
else info.secmax=inf;
end

if isfield(st,'nfmax'), info.nfmax=st.nfmax;
else info.nfmax=inf;
end

if isfield(st,'fbest'), info.fbest=st.fbest;
else info.fbest=0;
end

if isfield(st,'accf'), info.accf=st.accf;
else info.accf=1e-6;
end


info.initTime=cputime;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = x0;
[E,f] = fun(x);

info.res = norm(E);
info.iter = 0;
info.nf = 1;

% Initialize stopping information.
info.qso = norm(E);
info.sec = cputime-info.initTime;
info.done = 0;


if debug
    fprintf('[INIT] nf=%d  ||x||=%.16e  ||E||=%.16e  f=%.16e\n', ...
        info.nf,norm(x),norm(E),f);
    fprintf('[INIT] max|x|=%.16e  max|E|=%.16e\n', ...
        max(abs(x(:))),max(abs(E(:))));
end

% Method memory variables.
xold = x;
Eold = E;

debugPrevR = norm(E);
debugPrevF = f;

k=0;
while 1
    k=k+1;
    if k > tune.maxit, break; end
    info.done = 0;
    info.iter = k;

    switch upper(method)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'EG'   % Extragradient
        %
        % w_k = x_k - s E_k
        %
        % x_{k+1} = x_k - s E(w_k)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            w = x - s*E;

            [Ew,fw] = fun(w);

            info.nf = info.nf+1;

            % Check stopping criteria.
            sec       = (cputime-info.initTime);
            info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
            info.qso  = norm(Ew);
            info.done = (info.done)||(info.qso<=info.accf);
            info.sec  = sec;

            % Check stopping criteria.
            if ~info.done

                x = x - s*Ew;

                % Evaluate the residual at the new point.
                [E,f] = fun(x);
                r     = norm(E);

                info.res(end+1) = r;
                info.nf = info.nf+1;

                % Check stopping criteria.
                sec       = (cputime-info.initTime);
                info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
                info.qso  = r;
                info.done = (info.done)||(info.qso<=info.accf);
                info.sec  = sec;

            else

                x = w;
                E = Ew;
                f = fw;

            end

            % Check stopping criteria.
            if info.done, break; end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'OGDA' % Optimistic Gradient Descent-Ascent
        %
        % x_{k+1}
        % =
        % x_k - 2sE_k + sE_{k-1}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if k == 1

                x = x - s*E;

                % Evaluate the residual at the new point.
                [E,f] = fun(x);
                r = norm(E);

                info.res(end+1) = r;
                info.nf = info.nf+1;

                % Check stopping criteria.
                sec       = (cputime-info.initTime);
                info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
                info.qso  = r;
                info.done = (info.done)||(info.qso<=info.accf);
                info.sec  = sec;

                if debug && debugEvery>0 && ...
                        (mod(k-1,debugEvery)==0)

                    fprintf(['[OGDA k=%d nf=%d] FIRST STEP\n', ...
                             '   ||x||      = %.16e\n', ...
                             '   ||E||      = %.16e\n', ...
                             '   f          = %.16e\n', ...
                             '   max|x|     = %.16e\n', ...
                             '   max|E|     = %.16e\n', ...
                             '   growth(E)  = %.16e\n'], ...
                             k,info.nf,norm(x),r,f,max(abs(x(:))), ...
                             max(abs(E(:))),r/max(debugPrevR,realmin));
                end

                debugPrevR = r;
                debugPrevF = f;

                E0 = Eold;
                E1 = E;

            else

                x = x - 2*s*E1 + s*E0;

                % Evaluate the residual at the new point.
                [E2,f] = fun(x);
                r = norm(E2);

                info.res(end+1) = r;
                info.nf = info.nf+1;

                % Check stopping criteria.
                sec       = (cputime-info.initTime);
                info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
                info.qso  = r;
                info.done = (info.done)||(info.qso<=info.accf);
                info.sec  = sec;


                if debug && debugEvery>0 && ...
                        (mod(k-1,debugEvery)==0)

                    stepOGDA = -2*s*E1 + s*E0;

                    fprintf(['[OGDA k=%d nf=%d]\n', ...
                             '   ||x||          = %.16e\n', ...
                             '   ||E2||         = %.16e\n', ...
                             '   f              = %.16e\n', ...
                             '   max|x|         = %.16e\n', ...
                             '   max|E2|        = %.16e\n', ...
                             '   ||E1||         = %.16e\n', ...
                             '   ||E0||         = %.16e\n', ...
                             '   ||step||       = %.16e\n', ...
                             '   growth(E)      = %.16e\n', ...
                             '   growth(f)      = %.16e\n'], ...
                             k,info.nf,norm(x),r,f,max(abs(x(:))), ...
                             max(abs(E2(:))),norm(E1),norm(E0), ...
                             norm(stepOGDA), ...
                             r/max(debugPrevR,realmin), ...
                             f/max(debugPrevF,realmin));

                    if r > 2*debugPrevR
                        fprintf('   *** WARNING: residual more than doubled ***\n');
                    end

                    if f > 10*debugPrevF
                        fprintf('   *** WARNING: f increased by more than 10x ***\n');
                    end

                    if any(~isfinite(x(:))) || ...
                       any(~isfinite(E2(:))) || ~isfinite(f)

                        fprintf('   *** NONFINITE VALUE DETECTED ***\n');
                    end
                end

                debugPrevR = r;
                debugPrevF = f;

                % Update OGDA history.
                E0 = E1;
                E1 = E2;
                E  = E2;

            end

            % Check stopping criteria.
            if info.done, break; end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'EAG'  % Extra Anchored Gradient
        %
        % w_k =
        % x_k + theta_k(x0-x_k) - sE_k
        %
        % x_{k+1} =
        % x_k + theta_k(x0-x_k) - sE(w_k)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            theta = 1/(k+2);

            w = x + theta*(x0-x) - s*E;

            [Ew,fw] = fun(w);
            info.nf = info.nf+1;

            % Check stopping criteria.
            sec       = (cputime-info.initTime);
            info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
            info.qso  = norm(Ew);
            info.done = (info.done)||(info.qso<=info.accf);
            info.sec  = sec;

            % Check stopping criteria.
            if ~info.done

                x = x + theta*(x0-x) - s*Ew;

                % Evaluate the residual at the new point.
                [E,f] = fun(x);
                r = norm(E);

                info.res(end+1) = r;
                info.nf = info.nf+1;

                % Check stopping criteria.
                sec       = (cputime-info.initTime);
                info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
                info.qso  = r;
                info.done = (info.done)||(info.qso<=info.accf);
                info.sec  = sec;

            else

                x = w;
                E = Ew;
                f = fw;

            end

            % Check stopping criteria.
            if info.done, break; end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'FOGDA'   % Fast OGDA
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            alpha = 50;

            if k == 1

                x_prev = x;
                Ew_prev = E;

                x = x - s*E;

                [E,f] = fun(x);
                info.res(end+1) = norm(E);

                info.nf = info.nf + 1;

                % Check stopping criteria.
                sec       = (cputime-info.initTime);
                info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
                info.qso  = norm(E);
                info.done = (info.done)||(info.qso<=info.accf);
                info.sec  = sec;

                % Check stopping criteria.
                if info.done
                    break;
                end

            else

                theta = 1 - alpha/(k + alpha);
                c1    = alpha*s/(2*(k + alpha));
                c2    = (s/2)*(1 + k/(k + alpha));

                w = x + theta*(x - x_prev) - c1*Ew_prev;

                [Ew,fw] = fun(w);

                info.nf = info.nf + 1;

                % Check stopping criteria.
                sec       = (cputime-info.initTime);
                info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
                info.qso  = norm(Ew);
                info.done = (info.done)||(info.qso<=info.accf);
                info.sec  = sec;

                % Check stopping criteria.
                if info.done

                    x = w;
                    E = Ew;
                    f = fw;
                    break;

                end

                x_new = w - c2*(Ew - Ew_prev);

                x_prev  = x;
                x       = x_new;
                Ew_prev = Ew;

                [E,f] = fun(x);
                info.res(end+1) = norm(E);
                info.nf = info.nf + 1;

                % Check stopping criteria.
                sec       = (cputime-info.initTime);
                info.done = (sec>info.secmax)||(info.nf>=info.nfmax);
                info.qso  = norm(E);
                info.done = (info.done)||(info.qso<=info.accf);
                info.sec  = sec;

                % Check stopping criteria.
                if info.done
                    break;
                end

            end


        otherwise
            error('Unknown method');

    end


    if info.prt>=0
        disp(['norm of function at nf=',num2str(info.nf),' is ',...
            num2str(norm(E))]);
    end


    if debug && ~strcmpi(method,'OGDA') && ...
            debugEvery>0 && (mod(k-1,debugEvery)==0)

        rr = norm(E);
        ff = 0.5*(E(:)'*E(:));

        fprintf(['[%s k=%d nf=%d] ||x||=%.16e  ||E||=%.16e  ', ...
                 'f=%.16e  max|x|=%.16e  max|E|=%.16e\n'], ...
                 method,k,info.nf,norm(x),rr,ff, ...
                 max(abs(x(:))),max(abs(E(:))));
    end

end


if debug
    fprintf('\n================ FINAL DEBUG =================\n');
    fprintf('iterations = %d\n',info.iter);
    fprintf('nf         = %d\n',info.nf);
    fprintf('||x||      = %.16e\n',norm(x));
    fprintf('||E||      = %.16e\n',norm(E));
    fprintf('f          = %.16e\n',0.5*(E(:)'*E(:)));
    fprintf('max|x|     = %.16e\n',max(abs(x(:))));
    fprintf('max|E|     = %.16e\n',max(abs(E(:))));
    fprintf('==============================================\n\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final stopping information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = 0.5*norm(E)^2;
info.qso = norm(E);
info.sec = cputime-info.initTime;


if info.qso<=info.accf

    info.error='accuracy reached';
    disp('accuracy reached');

elseif info.nf>=info.nfmax

    info.error='nfmax reached';
    disp('nfmax reached');

elseif info.sec>=info.secmax

    info.error='secmax reached';
    disp('secmax reached');

else

    info.error='unknown';
    disp('unknown');

end

end


