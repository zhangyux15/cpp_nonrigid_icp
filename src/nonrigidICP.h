#pragma once
#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "model.h"
#include "KDTree.h"
#include "nodegraph.h"


class NonrigidICP
{
public:
	NonrigidICP() {}
	~NonrigidICP() = default;
	NonrigidICP(const NonrigidICP& _) = delete;
	NonrigidICP& operator=(const NonrigidICP& _) = delete;

	const Model& GetIterModel() const { return m_iterModel; }
	void SetWeight(const Eigen::VectorXf& wDeform, const float& wSmth, const float& wRegular);
	void SetModel(std::shared_ptr<const Model> srcModel, std::shared_ptr<const Model> tarModel);
	void SetNodeGraph(std::shared_ptr<const NodeGraph> nodeGraph);

	void Solve(const int& maxIterTime = 10, const float& updateThresh = 1e-5f, const std::string& debugPath="");
	
protected:
	std::shared_ptr<const Model> m_srcModel, m_tarModel;
	std::shared_ptr<const KDTree<float>> m_tarTree;
	std::shared_ptr<const NodeGraph> m_nodeGraph;
	Model m_iterModel;
	Eigen::Matrix4Xf m_warpField;
	Eigen::VectorXi m_corr;
	Eigen::MatrixXf m_deltaTwist;
	Eigen::VectorXf m_wDeform;
	float m_wSmth = 0.1f;
	float m_wRegular = 1e-3f; 
	float m_maxDist = 0.1f;
	float m_maxAngle = float(EIGEN_PI) / 4;

	void FindCorr();
	void UpdateWarpField();
	void UpdateModel();

	void CalcDeformTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXf& ATb);
	void CalcSmthTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXf& ATb);
};


