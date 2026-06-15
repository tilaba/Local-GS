/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <math_functions.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}



// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH_D2(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;
	result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;
	result = result +
	SH_C2[0] * xy * sh[4] +
	SH_C2[1] * yz * sh[5] +
	SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
	SH_C2[3] * xz * sh[7] +
	SH_C2[4] * (xx - yy) * sh[8];
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}


// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ __forceinline__ glm::vec3 computeColorFromSH_D3_M16(int idx, int deg, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * 16;
	glm::vec3 result = SH_C0 * sh[0];

	float x = dir.x;
	float y = dir.y;
	float z = dir.z;
	result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;
	result = result +
	SH_C2[0] * xy * sh[4] +
	SH_C2[1] * yz * sh[5] +
	SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
	SH_C2[3] * xz * sh[7] +
	SH_C2[4] * (xx - yy) * sh[8];
	
	result = result +
	SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
	SH_C3[1] * xy * z * sh[10] +
	SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
	SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
	SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
	SH_C3[5] * z * (xx - yy) * sh[14] +
	SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];

	result += 0.5f;
	return glm::max(result, 0.0f);
}


// Forward version of 2D covariance matrix computation
__device__ __forceinline__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}


// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D_OPT(float3 t, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void __forceinline__ computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA_Optimized(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum_opt(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// if (p_proj.x < -2.0f || p_proj.x > 2.0f || p_proj.y < -2.0f || p_proj.y > 2.0f)
    // {
    //     return;
    // }

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	float3 cov;
	if (cov3D_precomp != nullptr)
	{
		const float* cov3D = cov3D_precomp + idx * 6;
		cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		const float* cov3D = cov3Ds + idx * 6;
		cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	}

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det <= 0.0f)
		return;

	
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

    // 2. 算特征值得到精准像素半径 (删除了冗余的 max(lambda1, lambda2))
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + __fsqrt_rn(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * __fsqrt_rn(max(0.0f, lambda1)));

	if (my_radius < 1.0f)
    	return;

    // 3. 100% 安全的屏幕包围盒交叉测试
    if ((point_image.x + my_radius < 0.0f) || (point_image.x - my_radius > (float)W) ||
        (point_image.y + my_radius < 0.0f) || (point_image.y - my_radius > (float)H))
    {
        return; 
    }

	float opacity = opacities[idx];
    // -----------------------------------------------------------------
    float det_inv = __frcp_rn(det);
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
    float4 con_o = { conic.x, conic.y, conic.z, opacity};


  // Only counts tiles touched when nullptr is passed as array argment.
  uint32_t tiles_count = duplicateToTilesTouched(
      point_image, con_o, grid,
      0, 0, 0,
      nullptr, nullptr);
  if (tiles_count == 0)
    return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH_D3_M16(idx, D, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
	((float3*)(rgb + idx * C))[0] = {result.x, result.y, result.z};
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
    // radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = con_o;
  	tiles_touched[idx] = tiles_count;
}







// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
       // Compute extent in screen space (by finding eigenvalues of
       // 2D covariance matrix). Use extent to compute a bounding rectangle
       // of screen-space tiles that this Gaussian overlaps with. Quit if
       // rectangle covers 0 tiles. 
       float mid = 0.5f * (cov.x + cov.z);
       float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
       float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
       float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

  // Updated: Compute extent in screen space by identifying exact
  // screen-space tile.overlap with Gaussian.
  // No longer need radius
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	float4 con_o = { conic.x, conic.y, conic.z, opacities[idx] };
  // Only counts tiles touched when nullptr is passed as array argment.
  uint32_t tiles_count = duplicateToTilesTouched(
      point_image, con_o, grid,
      0, 0, 0,
      nullptr, nullptr);
  if (tiles_count == 0)
    return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
    // radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = con_o;
  	tiles_touched[idx] = tiles_count;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color, const float* depths = nullptr,
    float* __restrict__ invdepth = nullptr)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}


__device__ inline float2 getXIntervalAtY(const float4 con_o, float dy, float limit)
{
    // con_o = (-0.5a, b, -0.5c, w)  对应椭圆矩阵
    // dy = y - y_center
    // limit = -log(255 * w)

    // 椭圆方程：
    // E(x,y) = a x^2 + b x y + c y^2 <= limit
    float a = con_o.x;   // -0.5 * A
    float b = -con_o.y;  // b
    float c = con_o.z;   // -0.5 * C

    // Solve a x^2 + b x * dy + c * dy^2 = limit  --> ax^2 + bx + c = 0
    float discriminant = b * b * dy * dy - 4.0f * a * (c * dy * dy - limit);

    if (discriminant <= 0.0f)
        return {0.0f, 0.0f}; // no intersection, conservative

    float sqrtD = sqrtf(discriminant);
    float x1 = (-b * dy - sqrtD) / (2.0f * a);
    float x2 = (-b * dy + sqrtD) / (2.0f * a);

    return {x1, x2};
}


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_LOCAl_GS(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
	const float* depths = nullptr,
    float* __restrict__ invdepth = nullptr)
{
   // 基础索引计算
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y };
    uint2 pix = { pix_min.x + threadIdx.x, pix_min.y + threadIdx.y };

    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;
    
    // 线程与 Warp 索引
    uint32_t tid = threadIdx.y * BLOCK_X + threadIdx.x;
	uint32_t max_y = BLOCK_Y - 1;
	uint32_t max_x = BLOCK_X - 1;
    const uint32_t warp_id = tid / 32; // Block 内第几个 Warp (0-7)
	const uint32_t warp_bit = 1 << warp_id;
	const float LOG_ALPHA_THRESHOLD = log(1.0f/255.0f); 

    uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;
    // --- Shared Memory ---
    __shared__ float3  collected_feat[BLOCK_SIZE];
    __shared__ uint32_t collected_warp_mask[BLOCK_SIZE]; 
	__shared__ float4 collected_quad_const[BLOCK_SIZE]; // 存 A, B, C, 常数项
	__shared__ float2 collected_linear[BLOCK_SIZE];     // 存 一次项系数 x, y

    float T = 1.0f;
    float Color[CHANNELS] = {0.0f, 0.0f, 0.0f};
    uint32_t last_contributor = 0;
	float expected_invdepth = 0.0f;

	float local_x = (float)threadIdx.x; 
	float local_y = (float)threadIdx.y;
	float x_2 = local_x * local_x; // 最大也就 225，毫无精度压力
	float y_2 = local_y * local_y;
	float xy = local_x * local_y;

    // 遍历 Batch
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
		if (__syncthreads_and(T < 0.001f || done)) break;
		
		
        // --- 加载数据到 Shared Memory ---
        int progress = i * BLOCK_SIZE + tid;
        if (progress <  range.y - range.x)
        {	
            int coll_id = point_list[range.x + progress];
            float2 g_xy = points_xy_image[coll_id];
            float4 con_o = conic_opacity[coll_id];
            
            // collected_xy[tid] = g_xy;
            // collected_conic_opacity[tid] = {-0.5f * con_o.x, con_o.y, -0.5f * con_o.z, logf(con_o.w)};
			float cx = g_xy.x;
			float cy = g_xy.y;
			float A = -0.5f * con_o.x;
			float B = con_o.y;
			float C = -0.5f * con_o.z;
			float D = logf(con_o.w);
			
			// --- 终极优化：提前计算与像素无关的系数 ---
			float local_cx = cx - (float)pix_min.x; 
			float local_cy = cy - (float)pix_min.y;

			// 用局部坐标计算常数项和一次项
			float const_term = A * local_cx * local_cx - B * local_cx * local_cy + C * local_cy * local_cy + D;
			float lin_x = -2.0f * A * local_cx + B * local_cy;
			float lin_y = -2.0f * C * local_cy + B * local_cx;
			// 存入 Shared Memory
			collected_quad_const[tid] = {A, B, C, const_term};
			collected_linear[tid] = {lin_x, lin_y};

			
			// 加载特征 (这里以 CHANNELS=3 为例，可根据模板扩展)
			collected_feat[tid].x = features[coll_id * CHANNELS + 0];
			collected_feat[tid].y = features[coll_id * CHANNELS + 1];
			collected_feat[tid].z = features[coll_id * CHANNELS + 2];
			
			// // --- 计算 Warp 级掩码 (基于边界裁剪优化) ---
			float T0 = 2.0f * __logf(255.0f * con_o.w);

			float det = con_o.x * con_o.z - con_o.y * con_o.y;
			float inv_det = __fdividef(T0, det);

			float dx = __fsqrt_rn(fmaxf(0.0f, con_o.z * inv_det));
			float dy = __fsqrt_rn(fmaxf(0.0f, con_o.x * inv_det)) * 0.95f;

			float g_min_x = g_xy.x - dx;
			float g_max_x = g_xy.x + dx;
			float g_min_y = g_xy.y - dy;
			float g_max_y = g_xy.y + dy;

			// 裁剪到扫描线块 (16 像素宽)
			float pix_min_x = (float)pix_min.x;
			float pix_max_x = pix_min_x + max_x;
			float intersect_min_x = fmaxf(g_min_x, pix_min_x);
			float intersect_max_x = fminf(g_max_x, pix_max_x);

			// 预计算共用值
			float inv_C = __fdividef(1.0f, con_o.z);
			float dx_sq = dx * dx;                     // dx²
			float base_det = con_o.z * T0;

			float dx_clip_min = intersect_min_x - g_xy.x;
			float dx_clip_max = intersect_max_x - g_xy.x;

			float neg_fma_y = -con_o.y * dy;
			float scaled_clip_min = dx_clip_min * con_o.x;
			float scaled_clip_max = dx_clip_max * con_o.x;

			float dx_clip_min_sq = dx_clip_min * dx_clip_min;
			float dx_clip_max_sq = dx_clip_max * dx_clip_max;

			float det_prime_min = fmaxf(0.0f, base_det - det * dx_clip_min_sq);
			float det_prime_max = fmaxf(0.0f, base_det - det * dx_clip_max_sq);

			float dy_max = dy;

			if (neg_fma_y < scaled_clip_min) {
				dy_max = fmaf(-con_o.y, dx_clip_min, __fsqrt_rn(det_prime_min)) * inv_C;
			} else if (neg_fma_y > scaled_clip_max) {
				dy_max = fmaf(-con_o.y, dx_clip_max, __fsqrt_rn(det_prime_max)) * inv_C;
			}

			float pos_fma_y = con_o.y * dy;
			float dy_min = -dy;

			if (pos_fma_y < scaled_clip_min) {
				float inner = fmaf(con_o.y, dx_clip_min, __fsqrt_rn(det_prime_min));
				dy_min = -inner * inv_C;
			} else if (pos_fma_y > scaled_clip_max) {
				float inner = fmaf(con_o.y, dx_clip_max, __fsqrt_rn(det_prime_max));
				dy_min = -inner * inv_C;
			}

			// 转换为像素行索引
			float absolute_y_min = g_xy.y + dy_min;
			float absolute_y_max = g_xy.y + dy_max;

			int y_start_row = max((int)pix_min.y, __float2int_rd(absolute_y_min) + 1);

			int y_end_row   = min((int)(pix_min.y + max_y), __float2int_ru(absolute_y_max) - 1);

			uint32_t mask = 0;
			if (y_start_row <= y_end_row) {
				int pix_min_y_int = (int)pix_min.y;
				int w_start_exact = (y_start_row - pix_min_y_int) >> 1;
				int w_end_exact   = (y_end_row   - pix_min_y_int) >> 1;
				mask = ((1U << (w_end_exact - w_start_exact + 1)) - 1) << w_start_exact;
			}

			collected_warp_mask[tid] = mask;
        }
		int batch_size =  min(BLOCK_SIZE, toDo);
		block.sync();
		if (__all_sync(0xFFFFFFFF, T < 0.0001f || done)) continue;

		#pragma unroll 8
		for (int j = 0; j < batch_size; j++)
		{
			if (!(warp_bit & collected_warp_mask[j]))
    			continue;

			float4 quad_const = collected_quad_const[j];
			float2 linear = collected_linear[j];
			float3 f = collected_feat[j];

			float A = quad_const.x;
			float B = quad_const.y;
			float C = quad_const.z;
			float const_term = quad_const.w;

			float power = (A * x_2 + C * y_2 - B * xy) + (linear.x * local_x + linear.y * local_y) + const_term;
			float alpha = power > LOG_ALPHA_THRESHOLD ? fminf(0.99f, __expf(power)) : 0;
			float weighted_alpha = alpha * T;
			float next_T = T - weighted_alpha;
			Color[0] += f.x * weighted_alpha;
			Color[1] += f.y * weighted_alpha;
			Color[2] += f.z * weighted_alpha;

			last_contributor++;
			T = next_T;
		}
    }

    // --- 写回结果 ---
    if (inside)
    {
        for (int ch = 0; ch < CHANNELS; ch++) {
            out_color[ch * H * W + pix_id] = Color[ch] + T * bg_color[ch];
        }
    }
}





void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	dim3 grid_dim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1); // 恢复为 16
	dim3 block_dim(BLOCK_X, BLOCK_Y/2, 1);                               // 保持 16x8


	renderCUDA_LOCAl_GS<NUM_CHANNELS> << <grid, block>> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);

}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{

	preprocessCUDA_Optimized<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
