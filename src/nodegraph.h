#pragma once
#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "model.h"


struct NodeGraph
{
	Eigen::VectorXi nodeIdx;
	Eigen::MatrixXi nodeNet;
	Eigen::MatrixXi knn;
	Eigen::MatrixXf weight;

	NodeGraph() {};
	NodeGraph(const std::string& folder) { Load(folder); }
	void Load(const std::string& filename);
	void Save(const std::string& filename) const;
};


struct NodeGraphGenerator : public NodeGraph
{
	Eigen::VectorXf Dijkstra(const int& start, const Eigen::MatrixXf& w)const;
	void Generate(const Model& model);
	void CalcGeodesic(const Model& model);
	void SampleNode();
	void GenKnn();
	void GenNodeNet();
	void VisualizeNodeNet(const Model& model, const std::string& filename) const;
	void VisualizeKnn(const Model& model, const std::string& filename) const;
	void LoadGeodesic(const std::string& filename);
	void SaveGeodesic(const std::string& filename) const;
	
	int netDegree = 4;
	int knnSize = 4;
	float nodeSpacing = 0.03f;
	float cutRate = 5.f;
	std::vector<std::map<int, float>> geodesic;
};
