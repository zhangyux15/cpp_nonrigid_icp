#include "nonrigidICP.h"
#include "math_util.h"
#include <set>


void NonrigidICP::SetModel(std::shared_ptr<const Model> srcModel, std::shared_ptr<const Model> tarModel)
{
	m_srcModel = srcModel;
	m_tarModel = tarModel;
	m_tarTree = std::make_shared<KDTree<float>>(tarModel->vertices);
	m_iterModel = *m_srcModel;
	m_corr = Eigen::VectorXi::Constant(m_srcModel->vertices.cols(), -1);
}


void NonrigidICP::SetNodeGraph(std::shared_ptr<const NodeGraph> nodeGraph)
{
	m_nodeGraph = nodeGraph;
	m_warpField.resize(4, 4 * m_nodeGraph->nodeIdx.size());
	for (int i = 0; i < m_nodeGraph->nodeIdx.size(); i++)
		m_warpField.middleCols(4 * i, 4).setIdentity();
	m_deltaTwist.resize(6, m_nodeGraph->nodeIdx.size());
}


void NonrigidICP::SetWeight(const Eigen::VectorXf& wDeform, const float& wSmth, const float& wRegular)
{
	if (wDeform.size() == 0)
		m_wDeform = Eigen::VectorXf::Ones(m_srcModel->vertices.cols());
	else
		m_wDeform = wDeform;

	m_wSmth = wSmth;
	m_wRegular = wRegular;
}


void NonrigidICP::FindCorr()
{
	const float cosine = std::cosf(m_maxAngle);
	m_corr.setConstant(-1);
#pragma omp parallel for
	for (int sIdx = 0; sIdx < m_iterModel.vertices.cols(); sIdx++) {
		if (m_wDeform[sIdx] < FLT_EPSILON)
			continue;
		const Eigen::Vector3f v = m_iterModel.vertices.col(sIdx);
		const std::vector<std::pair<float, size_t>> nbors = m_tarTree->KNNSearch(v, 4);
		for (const auto& nbor : nbors) {
			if (nbor.first < m_maxDist) {
				if (m_iterModel.normals.col(sIdx).dot(m_tarModel->normals.col(nbor.second)) > cosine) {
					m_corr[sIdx] = int(nbor.second);
					break;
				}
			}
		}
	}

	std::cout << "corr:" << (m_corr.array() >= 0).count() << std::endl;
}


void NonrigidICP::UpdateWarpField()
{
#pragma omp parallel for
	for (int ni = 0; ni < m_deltaTwist.cols(); ni++)
		m_warpField.middleCols(4*ni,4) = MathUtil::Twist(m_deltaTwist.col(ni))* m_warpField.middleCols(4 * ni, 4);
}


void NonrigidICP::CalcDeformTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXf& ATb)
{
	if (m_wDeform.size() != m_srcModel->vertices.cols())
		m_wDeform = Eigen::VectorXf::Ones(m_srcModel->vertices.cols());

	std::vector<Eigen::Triplet<float>> triplets;
	Eigen::VectorXf b = Eigen::VectorXf::Zero(m_srcModel->vertices.cols());

	for (int sIdx = 0; sIdx < m_srcModel->vertices.cols(); sIdx++) {
		const int tIdx = m_corr[sIdx];
		const float wd = m_wDeform[sIdx];
		if (tIdx == -1 || wd < FLT_EPSILON)
			continue;

		const auto tn = m_tarModel->normals.col(tIdx);
		const auto iv = m_iterModel.vertices.col(sIdx);
		const auto tv = m_tarModel->vertices.col(tIdx);
		for (int i = 0; i < m_nodeGraph->knn.rows(); i++) {
			const int ni = m_nodeGraph->knn(i, sIdx);
			const float w = m_nodeGraph->weight(i, sIdx) * wd;
			if (w < FLT_EPSILON || ni == -1)
				continue;

			const int col = 6 * ni;
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col, w * (tn.z()*iv.y() - tn.y()*iv.z()))); // alpha_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 1, w * (tn.x()*iv.z() - tn.z()*iv.x()))); // beta_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 2, w * (tn.y()*iv.x() - tn.x()*iv.y()))); // gamma_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 3, w * tn.x())); // tx_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 4, w * tn.y())); // ty_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 5, w * tn.z())); // tz_ni
		}
		b[sIdx] = wd * (tn.dot(tv - iv));
	}
	Eigen::SparseMatrix<float> A(m_srcModel->vertices.cols(), m_nodeGraph->nodeIdx.size() * 6);
	A.setFromTriplets(triplets.begin(), triplets.end());
	const auto AT = A.transpose();
	ATA = AT * A;
	ATb = AT * b;
}


void NonrigidICP::CalcSmthTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXf& ATb)
{
	std::vector<Eigen::Triplet<float>> triplets;
	Eigen::VectorXf b = Eigen::VectorXf::Zero(3 * m_nodeGraph->nodeNet.size());
	for (int ni = 0; ni < m_nodeGraph->nodeIdx.size(); ni++) {
		const auto Ti = m_warpField.block<3, 4>(0, 4 * ni);
		for (int j = 0; j < m_nodeGraph->nodeNet.rows(); j++) {
			const int nj = m_nodeGraph->nodeNet(j, ni);
			if (nj == -1)
				continue;

			const auto sv = m_srcModel->vertices.col(m_nodeGraph->nodeIdx[nj]);
			const auto Tj = m_warpField.block<3, 4>(0, 4 * nj);
			const Eigen::Vector3f r = Ti * sv.homogeneous();
			const Eigen::Vector3f s = Tj * sv.homogeneous();
			const int row = 3 * (ni * int(m_nodeGraph->nodeNet.rows()) + j);
			const int coli = 6 * ni;
			const int colj = 6 * nj;

			// 1st row
			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 1, r.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 2, -r.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 3, 1.f));
			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 1, -s.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 2, s.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 3, -1.f));

			// 2nd row
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 0, -r.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 2, r.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 4, 1.f));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 0, s.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 2, -s.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 4, -1.f));

			// 3rd row
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 0, r.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 1, -r.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 5, 1.f));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 0, -s.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 1, s.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 5, -1.f));

			// bs
			b.segment<3>(row) = s - r;
		}
	}

	Eigen::SparseMatrix<float> A(3 * m_nodeGraph->nodeNet.size(), 6 * m_nodeGraph->nodeIdx.size());
	A.setFromTriplets(triplets.begin(), triplets.end());
	const auto AT = A.transpose();
	ATA = m_wSmth * AT * A;
	ATb = m_wSmth * AT * b;
}


void NonrigidICP::Solve(const int& maxIterTime, const float& updateThresh, const std::string& debugPath)
{
	Eigen::SparseMatrix<float> ATA, ATAs;
	Eigen::VectorXf ATb, ATbs;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++) {
		FindCorr();
		CalcDeformTerm(ATA, ATb);
		CalcSmthTerm(ATAs, ATbs);
		Eigen::SparseMatrix<float> ATAr(6 * m_nodeGraph->nodeIdx.size(), 6 * m_nodeGraph->nodeIdx.size());
		ATAr.setIdentity();
		ATAr *= m_wRegular;
		ATA += ATAs+ATAr;
		ATb += ATbs;

		Eigen::Map<Eigen::VectorXf>(m_deltaTwist.data(), m_deltaTwist.size()) = solver.compute(ATA).solve(ATb);
		assert(solver.info() == 0);

		if (m_deltaTwist.norm() < updateThresh)
			break;

		// debug 
		std::cout << "delta twist: " << m_deltaTwist.norm() << std::endl;
		UpdateWarpField();
		UpdateModel();

		if (debugPath != "") {
			static int cnt = 0;
			m_iterModel.Save(debugPath + "/" + std::to_string(cnt++) + ".obj");
		}
	}
}


void NonrigidICP::UpdateModel()
{
	for (int sIdx = 0; sIdx < m_srcModel->vertices.cols(); sIdx++) {
		Eigen::Matrix4f T = Eigen::Matrix4f::Zero();
		
		for (int i = 0; i < m_nodeGraph->knn.rows(); i++) {
			const int ni = m_nodeGraph->knn(i, sIdx);
			if (ni != -1)
				T += m_nodeGraph->weight(i, sIdx) * m_warpField.middleCols(4 * ni, 4);
		}

		m_iterModel.vertices.col(sIdx) = T.topLeftCorner(3, 4) * m_srcModel->vertices.col(sIdx).homogeneous();
		m_iterModel.normals.col(sIdx) = T.topLeftCorner(3, 3) * m_srcModel->normals.col(sIdx);
	}
}

