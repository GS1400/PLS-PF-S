function [x,f,info]=DFSdir(fun,x,st,tune)

% DFSDIR  PLS-S/PF-S methods for unconstrained monotone equations.
%
% The routine solves E(x)=0 on R^n using the Jacobian-free subspace
% directions and projection framework considered in the paper.

% Check the operator handle.
if isempty(fun)
  message = 'CGLS needs the function handle fun to be defined';
  disp(message)
  return
elseif ~isa(fun,'function_handle')
  message = 'fun should be a function handle';
  disp(message)
  return
end

if ~exist('tune','var'), tune=[]; end


if ~isfield(tune,'ro'), tune.ro=0.5; end
ro=tune.ro;

if ~isfield(tune,'eta'), tune.eta=1e-2; end
eta=tune.eta;

if ~isfield(tune,'mu'), tune.mu=0.01; end
mu=tune.mu;

if ~isfield(tune,'zeta'), tune.zeta=1e-3; end
zeta=tune.zeta;

if ~isfield(tune,'projType'), tune.projType='exact'; end

if ~isfield(tune,'Deltaangle'), tune.Deltaangle=9e-1; end

% Lipschitz constant used in the line-search and fixed-step safeguards.
if ~isfield(tune,'L')
    error('The Lipschitz constant tune.L must be provided.');
end
L=tune.L;

% Backtracking factor gamma.
gamma=1/ro;

% Lower safeguard for the line-search step.
mumin=min(mu,tune.Deltaangle/(gamma*(L+eta)));

% Select the subspace-direction formula.
if ~isfield(tune,'state'), tune.state='FR'; end
state=tune.state;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLS-S and PF-S variants
%
% The base state names select the projected-line-search realization PLS-S:
%
% FR, PR, HS, DY, LS, HZ, DL,
% 3PR, 3HS, An1, An2, De, A1,...,A8.
%
% Appending '-F' selects the corresponding fixed-step realization PF-S:
%
% FR-F, PR-F, HS-F, DY-F, LS-F, HZ-F, DL-F,
% 3PR-F, 3HS-F, An1-F, An2-F, De-F, A1-F,...,A8-F.
%
% For PF-S, the fixed step satisfies
%
%       0 < alpha <= min{1,Delta_angle/(L+rho)}.
%
% In the present implementation eta corresponds to rho.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fixedStep=false;

if length(state)>2 && strcmp(state(end-1:end),'-F')

    fixedStep=true;

    % Remove the PF-S suffix; the direction formula is unchanged.
    state=state(1:end-2);

end

if fixedStep
    tune.Deltaangle=9e-1;
end
% Largest admissible fixed step.
alphaMax=min(1,tune.Deltaangle/(L+eta));

if fixedStep

    if isfield(tune,'alpha')
        alpha=tune.alpha;
    else
        alpha=alphaMax;
    end

    if ~(isscalar(alpha) && isfinite(alpha) && ...
         alpha>0 && alpha<=alphaMax)

        error(['For a fixed-step PLS-S variant, tune.alpha must satisfy ',...
               '0 < alpha <= min(1,Deltaangle/(L+eta)).']);

    end

end


if ~exist('st','var'), st=[]; end


% Performance and stopping information
%  .prt          printing level
%  .secmax       maximum CPU time
%  .nfmax        maximum number of operator evaluations
%  .fbest        target merit value
%  .accf         residual stopping tolerance
%  .qso          current residual norm
%  .initTime     initial CPU time
%  .done         stopping flag

% Printing level.
if isfield(st,'prt'), info.prt=st.prt;
else, info.prt=-1;
end

% Stopping criteria.
if isfield(st,'secmax'), info.secmax=st.secmax;
else, info.secmax=inf;
end

if isfield(st,'nfmax'), info.nfmax=st.nfmax;
else, info.nfmax=inf;
end

if isfield(st,'fbest'), info.fbest=st.fbest;
else, info.fbest=0;
end

if isfield(st,'accf'), info.accf=st.accf;
else, info.accf=1e-6;
end


info.initTime=cputime;

x=x(:);


iter=0;
info.nf=0;


[E,f]=fun(x);
info.nf=info.nf+1;

% Initial subspace and update directions coincide.
p=-E;
d=p;

info.done=0;
info.qso=norm(E);
info.sec=0;

xold=x;
Eold=E;


% Initial stopping test.
if info.qso<=info.accf
    info.done=1;
end


% Main iteration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while ~info.done

    iter=iter+1;

    % Preserve the preceding subspace and update directions:
    % pold = p_{ell-1}
    % dold = p_{ell-1}^{scal}
    pold=p;
    dold=d;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step-size selection
    %
    % fixedStep = false: projected line search (PLS-S).
    % fixedStep = true : fixed step alpha (PF-S).
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if fixedStep

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % PF-S fixed step
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        j=0;
        mutrial=alpha;

        xtrial=x+mutrial*dold;
        Etrial=fun(xtrial);
        info.nf=info.nf+1;

        % Check the evaluation/time budget.
        sec=(cputime-info.initTime);
        info.done=(sec>info.secmax)||(info.nf>=info.nfmax);
        info.sec=sec;

        if info.prt>=0
            disp(['norm of trial residual at nf=',num2str(info.nf),' is ',...
                num2str(norm(Etrial))]);
        end


    else

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % PLS-S projected line search
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        j=0;
        mutrial=mu;

        xtrial=x+mutrial*dold;
        Etrial=fun(xtrial);
        info.nf=info.nf+1;

        % Check the evaluation/time budget.
        sec=(cputime-info.initTime);
        info.done=(sec>info.secmax)||(info.nf>=info.nfmax);
        info.sec=sec;

        if info.prt>=0
            disp(['norm of trial residual at nf=',num2str(info.nf),' is ',...
                num2str(norm(Etrial))]);
        end

        if info.done
            break
        end


        % Backtracking line search.
        while (Etrial'*dold) > (-eta*mutrial*(norm(dold))^2)

            if mutrial<=mumin
                % Apply the lower step-size safeguard.
                mutrial=mumin;
                xtrial=x+mutrial*dold;
                Etrial=fun(xtrial);
                info.nf=info.nf+1;

                if (Etrial'*dold) > (-eta*mutrial*(norm(dold))^2)
                    warning('Line-search condition not satisfied at mu_min.');
                end
                break
            end

            j=j+1;
            mutrial=max(mutrial/gamma,mumin);

            xtrial=x+mutrial*dold;
            Etrial=fun(xtrial);
            info.nf=info.nf+1;

            sec=(cputime-info.initTime);
            info.done=(sec>info.secmax)||(info.nf>=info.nfmax);
            info.sec=sec;

            if info.prt>=0
                disp(['norm of trial residual at nf=',num2str(info.nf),' is ',...
                    num2str(norm(Etrial))]);
            end

            if info.done
                break
            end
        end

    end


    if info.done
        break
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Exact projection onto H_{ell-1}^-
    %
    % H_{ell-1}^- =
    % {z : (z-xtrial)'*Etrial <= 0}
    %
    % x_ell = P_{H_{ell-1}^-}(xold)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nEtrial2=Etrial'*Etrial;

    if nEtrial2==0
        x=xtrial;
        E=Etrial;
        info.qso=0;
        info.done=1;
        break
    end

    lambda=((xold-xtrial)'*Etrial)/nEtrial2;
    x=xold-lambda*Etrial;

    E=fun(x);
    info.nf=info.nf+1;

    info.qso=norm(E);

    sec=(cputime-info.initTime);
    info.done=(sec>info.secmax)||(info.nf>=info.nfmax)||...
              (info.qso<=info.accf);
    info.sec=sec;

    if info.prt>=0
        disp(['norm of function at nf=',num2str(info.nf),' is ',...
            num2str(info.qso)]);
    end

    if info.done
        break
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Subspace quantities:
    %
    % y_{ell-1}=E(bar{x}_{ell-1})-E(x_{ell-1})
    %
    % s_{ell-1}=bar{x}_{ell-1}-x_{ell-1}
    %          =mu_{ell-1}p_{ell-1}^{scal}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    y=Etrial-Eold;
    s=xtrial-xold;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Curvature-denominator safeguards.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    eps_curv=1e-12;
    phi_saf=eps_curv;
    psi_saf=eps_curv;

    ss=s'*s;

    if ss>0 && abs(s'*y)<=eps_curv
        y=y+phi_saf*s;
    end

    if ss>0 && abs(y'*pold)<=eps_curv
        u=pold-(s'*pold/ss)*s;
        nu=norm(u);

        if nu>0
            y=y+psi_saf*norm(s)*(u/nu);
        end
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Jacobian-free subspace direction p_ell
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    switch state


        %========================================================
        % JFS2 family
        %========================================================

        case 'FR'

            denom=Eold'*Eold;

            if denom>0
                betak=(E'*E)/denom;

                if isfinite(betak)
                    p=-E+betak*pold;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        case 'PR'

            denom=Eold'*Eold;

            if denom>0
                betak=(y'*E)/denom;

                if isfinite(betak)
                    p=-E+betak*pold;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        case 'HS'

            denom=y'*pold;

            if abs(denom)>eps
                betak=(y'*E)/denom;

                if isfinite(betak)
                    p=-E+betak*pold;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        case 'DY'

            denom=y'*pold;

            if abs(denom)>eps
                betak=(E'*E)/denom;

                if isfinite(betak)
                    p=-E+betak*pold;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        case 'LS'

            denom=Eold'*pold;

            if abs(denom)>eps
                betak=-(E'*y)/denom;

                if isfinite(betak)
                    p=-E+betak*pold;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        case 'DL'

            denom=y'*pold;

            if abs(denom)>eps

                % Modified vector used in the DL coefficient.
                ybar=y-2*(norm(y)^2/denom)*pold;

                betaDL=(E'*ybar)/denom;

                if isfinite(betaDL)
                    p=-E+betaDL*pold;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        case 'HZ'

            denom=y'*pold;

            if abs(denom)>eps

                % Modified vector used in the HZ coefficient.
                tHZ=0.1;
                ytilde=y-tHZ*pold;

                betaHZ=(E'*ytilde)/denom;

                if isfinite(betaHZ)
                    p=-E+betaHZ*pold;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        %========================================================
        % JFS3dir1-Z1
        %========================================================

        case '3PR'

            denom=Eold'*Eold;

            if denom>0

                beta1=(y'*E)/denom;
                beta2=-(pold'*E)/denom;

                if isfinite(beta1) && isfinite(beta2)
                    p=-E+beta1*pold+beta2*y;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        %========================================================
        % JFS3dir1-Z2
        %========================================================

        case '3HS'

            denom=y'*pold;

            if abs(denom)>eps

                beta1=(y'*E)/denom;
                beta2=-(pold'*E)/denom;

                if isfinite(beta1) && isfinite(beta2)
                    p=-E+beta1*pold+beta2*y;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        %========================================================
        % JFS3dir2-An with j=1
        %
        % beta3 = -s'E/(y's)
        % beta4 = (1+phi) beta3 + y'E/(y's)
        % phi   = ||y||^2/(y's)
        %
        % p = -E + beta3*y + beta4*s
        %========================================================

        case 'An1'

            ys=y'*s;
            ss=s'*s;

            if abs(ys)>eps

                yy=y'*y;
                yF=y'*E;
                sF=s'*E;

                phi=yy/ys;

                beta3=-sF/ys;
                beta4=(1+phi)*beta3+yF/ys;

                if isfinite(beta3) && isfinite(beta4)
                    p=-E+beta3*y+beta4*s;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        %========================================================
        % JFS3dir2-An with j=2
        %
        % beta4 = (1+2 phi) beta3 + y'E/(y's)
        %========================================================

        case 'An2'

            ys=y'*s;
            ss=s'*s;

            if abs(ys)>eps

                yy=y'*y;
                yF=y'*E;
                sF=s'*E;

                phi=yy/ys;

                beta3=-sF/ys;
                beta4=(1+2*phi)*beta3+yF/ys;

                if isfinite(beta3) && isfinite(beta4)
                    p=-E+beta3*y+beta4*s;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        %========================================================
        % JFS3dir2-De
        %
        % phi2 = 1-min{1,phi}
        % beta3=-s'E/(y's)
        % beta4=phi2 beta3+y'E/(y's)
        %========================================================

        case 'De'

            ys=y'*s;
            ss=s'*s;

            if abs(ys)>eps

                yy=y'*y;
                yF=y'*E;
                sF=s'*E;

                phi=yy/ys;
                phi2=1-min(1,phi);

                beta3=-sF/ys;
                beta4=phi2*beta3+yF/ys;

                if isfinite(beta3) && isfinite(beta4)
                    p=-E+beta3*y+beta4*s;
                else
                    p=-E;
                end
            else
                p=-E;
            end


        %========================================================
        % JFS3dir3-A1
        %========================================================

        case 'A1'

            [p,ok]=JFS3dir3(E,Eold,y,pold,1);

            if ~ok
                p=-E;
            end


        %========================================================
        % JFS3dir3-A2
        %========================================================

        case 'A2'

            [p,ok]=JFS3dir3(E,Eold,y,pold,2);

            if ~ok
                p=-E;
            end


        %========================================================
        % JFS3dir3-A3
        %========================================================

        case 'A3'

            [p,ok]=JFS3dir3(E,Eold,y,pold,3);

            if ~ok
                p=-E;
            end


        %========================================================
        % JFS3dir3-A4
        %========================================================

        case 'A4'

            [p,ok]=JFS3dir3(E,Eold,y,pold,4);

            if ~ok
                p=-E;
            end


        %========================================================
        % JFS3dir3-A5
        %========================================================

        case 'A5'

            [p,ok]=JFS3dir3(E,Eold,y,pold,5);

            if ~ok
                p=-E;
            end


        %========================================================
        % JFS3dir3-A6
        %========================================================

        case 'A6'

            [p,ok]=JFS3dir3(E,Eold,y,pold,6);

            if ~ok
                p=-E;
            end


        %========================================================
        % JFS3dir3-A7
        %========================================================

        case 'A7'

            [p,ok]=JFS3dir3(E,Eold,y,pold,7);

            if ~ok
                p=-E;
            end


        %========================================================
        % JFS3dir3-A8
        %========================================================

        case 'A8'

            [p,ok]=JFS3dir3(E,Eold,y,pold,8);

            if ~ok
                p=-E;
            end


        otherwise

            warning('Unknown subspace direction. Falling back to -E.');
            p=-E;

    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Safeguard for a vanishing or nonfinite direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nF=norm(E);
    np=norm(p);

    if np==0 || ~isfinite(np)
        p=-E;
        np=nF;
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Norm safeguard applied before the angle test:
    %
    % zeta ||E|| <= ||d|| <= ||E||,
    %
    % with d representing the scaled update direction.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    sc=median([zeta*nF np nF])/np;
    d=sc*p;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Angle safeguard
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dTF=E'*d;

    if isfinite(dTF)

        sigma1=E'*E; sigma2=d'*d; sigma12=sigma1*sigma2;
        sigmanew=dTF/sqrt(sigma12);

        angleOk=(sigmanew<=-tune.Deltaangle);

        if angleOk
           % The current direction satisfies the angle condition.
%            if info.prt>=1,
%                disp('the angle condition holds');
%            end
        else % The angle condition is violated.
%            if info.prt>=1,
%                disp('enforce the angle condition');
%            end
           w=(sigma12*max(0,1-sigmanew^2))/(1-tune.Deltaangle^2);
           t=(dTF+tune.Deltaangle*sqrt(w))/sigma1;
           wtOk=(w>0 && isfinite(t));

           if wtOk
               d=d-t*E;

               % Store the angle-corrected direction.
               p=d;

               % Reapply the norm safeguard after the angle correction.
               np=norm(p);

               if np==0 || ~isfinite(np)
                   p=-E;
                   np=nF;
               end

               sc=median([zeta*nF np nF])/np;
               d=sc*p;

           else
               % Use -E in the degenerate collinear case.
               p=-E;
               d=-E;
           end
        end
    else
        p=-E;
        d=-E;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Store the iterate and residual for the next update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    xold=x;
    Eold=E;

end


f=0.5*norm(E)^2;
info.iter=iter;
info.qso=norm(E);
info.sec=cputime-info.initTime;


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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% JFS3dir3 family
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [p,ok]=JFS3dir3(E,Eold,y,pold,type)

p=-E;
ok=false;


% Denominator associated with the preceding direction.
yTp=y'*pold;

if abs(yTp)<=eps || norm(pold)==0 || norm(Eold)==0
    return
end


% Modified vector used in beta_l.
ybar=y-(2*norm(y)^2/yTp)*pold;

beta=(E'*ybar)/yTp;

if ~isfinite(beta)
    return
end


% Normalized residual-direction correlation.
Re=(E'*pold)/(norm(Eold)*norm(pold));

if ~isfinite(Re)
    return
end


% Auxiliary vector h_l.
h=Eold;

% Ensure E(x_l)'h_l is numerically nonzero.
if abs(E'*h)<=eps*max(1,norm(E)*norm(h))
    h=E;
end

Eh=E'*h;

if abs(Eh)<=eps
    return
end


% Parameters of the JFS3dir3 family.
gammabar=0.8;
varsigma1=0.01;
varsigma2=100;


switch type

    case 1
        gammabarell=1-gammabar*abs(beta*Re);

    case 2
        gammabarell=1+gammabar*abs(beta*Re);

    case 3
        gammabarell=1-gammabar*beta*Re;

    case 4
        gammabarell=1+gammabar*beta*Re;

    case 5
        gammabarell=1-gammabar*abs(Re);

    case 6
        gammabarell=1+gammabar*abs(Re);

    case 7
        gammabarell=1-gammabar*Re;

    case 8
        gammabarell=1+gammabar*Re;

    otherwise
        return

end


gammaell=max(varsigma1,min(varsigma2,gammabarell));


beta5=...
    -((gammaell-1)*(norm(E)^2)+beta*(E'*pold))/Eh;


if ~isfinite(beta5)
    return
end


p=-E+beta*pold+beta5*h;

ok=all(isfinite(p));

end
