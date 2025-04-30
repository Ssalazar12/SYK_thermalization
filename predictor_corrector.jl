using LinearAlgebra 
using NPZ
using Printf
using HDF5
using Distributed

# ------------------------------------------------------------------------------------------
# INITIALIZE SYSTEM PARAMETERS
# ------------------------------------------------------------------------------------------

Num_points = 3000
Sigma_ = 0.74
Origin_index = 1500
Delta_t = 0.06
Jq_magnitute_pre = 1.0 
Jq_magnitute_post = 0.0
J2_magnitude_post = 0.01

"""
j2_list = [0.09,0.01,0.05] # kinetic coupling
nt_list = repeat([Num_points], length(j2_list)) # size of the matrix holdin the g(t1,t2)
σ_list = repeat([Sigma_], length(j2_list)) # integration constant sets initial temperature
origin_list = repeat([Origin_index], length(j2_list))# index where the t=0 is located
dt_list= repeat([Delta_t], length(j2_list)) # discretization time
J_preq_list = repeat([Jq_magnitute_pre], length(j2_list)) # interaction coupling before quench
J_list = repeat([Jq_magnitute_post], length(j2_list)) # interaction coupling after quench
"""

σ_list = [0.15, 0.3, 0.74, 0.95]
current_list = σ_list

j2_list = repeat([J2_magnitude_post], length(current_list)) # kinetic coupling
nt_list = repeat([Num_points], length(current_list)) # size of the matrix holdin the g(t1,t2)
# σ_list = repeat([Sigma_], length(current_list)) # integration constant sets initial temperature
origin_list = repeat([Origin_index], length(current_list))# index where the t=0 is located
dt_list= repeat([Delta_t], length(current_list)) # discretization time
J_preq_list = repeat([Jq_magnitute_pre], length(current_list)) # interaction coupling before quench
J_list = repeat([Jq_magnitute_post], length(current_list)) # interaction coupling after quench

# ------------------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------------------

# creates a grid of xy values for evaluating an plotting
function meshgrid(xin::LinRange,yin::LinRange)
    nx=length(xin)
    ny=length(yin)
    xout=zeros(ny,nx)
    yout=zeros(ny,nx)
    for jx=1:nx
        for ix=1:ny
            xout[ix,jx]=xin[jx]
            yout[ix,jx]=yin[ix]
        end
    end
    return (x=xout, y=yout)
  end
  

function eq_g!(ret::Matrix{ComplexF64}, t1::Array{Float64}, t2::Array{Float64},σ::Float64,j::Float64)
    # calculates the equuilibrium solutions for times t1 and t2. notice how we modify the value of ret
    # ret is the matrix that is being modified
    # t1,t2: arrays or scalars containig the coordinates
    # σ, J: scalars that set the parameters of the sysyem
    θ = acos(σ/j)
    ret .= @. 2 *log( σ/(j *cosh( θ *1im + σ *(t1 -t2))))
    # indicate that it does not return anything
    nothing 
end


function create_G(T1::Matrix{Float64} ,T2::Matrix{Float64} ,step_number::Int64, origin_index::Int64, σ::Float64, j::Float64)
    # initializes the green's function
    # T1,T2: arrays that represent the meshgrid of the t1-t2 plane
    # σ, J: scalars that set the parameters of the sysyem
    # step_number: the size of the matrix
    # origin_index: indicates the position of the origin in t1,t2 plane
    # j: indicates the interaction strength prequench 
    
    G = complex(zeros(step_number,step_number))
    # fill all the quadrants and then remove the post quench part
    eq_g!(G,T1,T2, σ, j)
    # quadB and A
    G[1:end, origin_index+1:end] .= NaN +NaN*im # 0.0 + 0.0im
    # quadD and A
    G[origin_index+1:end, 1:end] .= NaN +NaN*im # 0.0 + 0.0im
    G[diagind(G)].= 0.0+ 0.0im 
    
    return G
end


function integrate_trapz(y::SubArray{ComplexF64}, Δ::Float64)
    # y:  the function to be integrated
    # Δ: the discretization step 
        
    integral = sum(y[1:end-1] + y[2:end]) * Δ/ 2
    
    return integral
end

function RHS_1(g::SubArray{ComplexF64}, jq::SubArray{Float64}, j2::SubArray{Float64}, tn::Int64, Δ::Float64)
    # calculates the right hand side of the kadanoff-baym equations for the predictor corrector method
    # g: slice of the two-time green's function at the current tn timestep  
    # tn: curreent horizontal time step
    # tm: max vertical time steo
    # jq: the function for the interaction coupling
    # j2: function for the kinetic coupling

    # calculate the first integral up to tn
    t3 = 1:tn
    integrand = @. @views  - jq[tn]*jq[t3]*( exp(g[t3]) + exp(conj(g[t3])) ) - 2*j2[tn]*j2[t3]
    ∫1 = integrate_trapz(@views(integrand[t3]), Δ)

    return ∫1 
end

function RHS_2(g::SubArray{ComplexF64}, jq::SubArray{Float64}, j2::SubArray{Float64}, tn::Int64, tm::Int64, Δ::Float64 )
    # calculates the right hand side of the kadanoff-baym equations for the predictor corrector method
    # g: slice of the two-time green's function at the current tn timestep  
    # tn: curreent horizontal time step
    # tm: max vertical time steo
    # jq: the function for the interaction coupling
    # j2: function for the kinetic coupling

    # calculate the second integral up to tm
    t3 = 1:tm
    integrand = @. @views   2*jq[tn]*jq[t3]*exp(g[t3]) + 2*j2[tn]*j2[t3]
    ∫2 = integrate_trapz(@views(integrand[t3]), Δ)

    return ∫2
end


# ------------------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------------

println("Num threads: ", Threads.nthreads())

# initialize multithreading

Threads.@threads for index_ in 1:length(j2_list)

    Nt::Int64  = nt_list[index_]
    origin_i::Int64 = origin_list[index_]
    dt::Float64 = dt_list[index_]
    J2_magnitude::Float64  = j2_list[index_]
    σ::Float64  = σ_list[index_]
    J_magnitude::Float64 = J_list[index_]
    J_preq::Float64 = J_preq_list[index_]

    println(Nt," " ,origin_i, " " ,dt, " " , " ", J2_magnitude, " ",σ, " " ,J_magnitude)

    # make a time range with the origin at the center
    time_range = LinRange(-origin_i*dt, (Nt-origin_i)*dt, Nt);
    tmin = minimum(time_range)

    # value for the quench
    J2 = zeros(Nt)
    J2[origin_i:end] .= J2_magnitude 

    J = ones(Nt)
    J[1:origin_i] .= J_preq
    J[origin_i:end] .= J_magnitude 

    print("Min and maxt: \n")
    print(minimum(time_range)," ", maximum(time_range), "\n")

    print("initial βJ: \n")
    β = 2.0*acos(σ/J_preq)*J_preq*(1/σ)
    print(β, '\n')
    print("Starting Solver ... \n")

    # make a grid of points and initialize the green's function
    t1, t2 = meshgrid(time_range, time_range);
    G_calc = create_G(t1,t2,Nt, origin_i, σ, J_preq);

    # create the file name for saving into h5
    mint = minimum(time_range)
    maxt = maximum(time_range)
    str_file_name = @sprintf("datasets/pred_sigma%f_Jtwo%f_J%f_dt%f_origin%i_mint%f_maxt%f.h5", 
                              σ, J2_magnitude, J_magnitude, dt, origin_i,mint,maxt)
     # this is for the iteration tracker
    str_text= @sprintf("iterations_Jtwo%f.txt", J2_magnitude)

    # iterate over horizontal time
    for n in origin_i:(Nt-1)
        # iterate over vertical time for the prediction
        if(n%50==0)
            fil = open(str_text,"a")
            write(fil, "loops elapsed: ", string(n), '\n')
            close(fil)
        end

        # initialize the predictor for n+1 diagonal is given by initial condition
        g_pred = complex(zeros(n+1))
        # so we can save the values for the corrector
        F1_vector = complex(zeros(n))

        # first integral up until tn is the same for all m 
        integral_tn = RHS_1(@view(G_calc[1:n,n]), @views(J[1:n]), @views(J2[1:n]), n, dt) 

        for m in 1:n   
            # calculate the RHS of the KB equations at (tn,tm)
            integral_tm = RHS_2(@view(G_calc[1:n,n]), @views(J[1:n]), @views(J2[1:n]), n, m, dt) 
            F1 = integral_tn + integral_tm
            F1_vector[m] = F1 # save  for future use
            # calculate the predictor
            g_pred[m] = G_calc[m,n] + dt*F1
        end
        # iterate for the corrector
        integral_tn = RHS_1(@view(g_pred[1:n+1]), @views(J[1:n+1]), @views(J2[1:n+1]), n+1, dt) 
        for m in 1:n
            # remember that here we need n+1
            integral_tm = RHS_2(@view(g_pred[1:n+1]), @views(J[1:n+1]), @views(J2[1:n+1]), n+1, m, dt) 
            F2 = integral_tn + integral_tm
            G_calc[m,n+1] = G_calc[m,n] + 0.5*dt*( F1_vector[m] + F2) 
            G_calc[n+1,m] = conj(G_calc[m,n+1])
        end
        
       # check for nans in the current iterations and break the loop if we find any
        if any(isnan, G_calc[1:n,n]) == true
            print("Found Nan in calculation at \n")
            print("t = ", (n-origin_i)*dt , "iteration = ", n , "J2 = ", J2_magnitude, "\n")
            print("stopping procedure ... \n")
            
            # now open the text file and print the Nan business
            fil = open(str_text,"a")
                write(fil, "FOUND NAN AT ITERATION ", string(n), '\n')
            close(fil)
            
            break
        end   
        # save checkpoints
        if(n%500==0)
            println("Saving Checkpoint up to ", n)
            save_up_to = size(G_calc[1:n,1:n])[1]
            fid = h5open(str_file_name, "w")
            fid["values",chunk=(save_up_to,1), compress=3] = G_calc[1:n,1:n]
            close(fid)
        end
        
    end

    # save the final state
    fid = h5open(str_file_name, "w")
    fid["values",chunk=(Nt,1), compress=3] = G_calc
    close(fid)

    print(str_file_name)

end







