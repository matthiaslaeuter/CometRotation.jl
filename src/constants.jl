const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}

module Constants

using StaticArrays

const AU_M       = 149597870700.0      
const m2au       = 1.0 / AU_M

const au2dist    = 1.0e3    
const day2time   = 1.0e-1   
const kg2mass    = 1.0e-9    
const m2dist     = m2au * au2dist
const dist2m     = 1.0 / m2dist

const avogadro   = 6.02214076e23       
const h2o_kg_mol = 0.0180153           
const molec_kg   = h2o_kg_mol / avogadro   
const solarconst_w_m2 = 1361.0                 
const sun_gm_au3_day2 = 2.9591220828411956e-4
const cg_kg      = 1.0e13                  
const cg_mass    = cg_kg * kg2mass
const jd_perihel_tk = 2457247.545 * day2time  

const day2sec    = 86400.0
const sec2day    = 1.0 / day2sec
const time2day   = 1.0 / day2time
const min2sec    = 60.0
const dist2au    = 1.0 / au2dist
const sec2time   = sec2day * day2time
const time2sec   = 1.0 / sec2time
const mass2kg    = 1.0 / kg2mass

const GM_SUN_SI = sun_gm_au3_day2 * AU_M^3 / day2sec^2
const R0_SI = SVector(1.659709691844e+11, -4.665730423448e+11, -2.654717135282e+11)
const V0_SI = SVector(7.998579476298e+03,  1.123208398331e+04,  5.107307193809e+03)
const reference_jd     = jd_perihel_tk - 400 * day2time

function day2jd(day::Float64)
  return jd_perihel_tk + day * day2time
end

end
