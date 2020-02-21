#include "nodegraph.h"
#include "math_util.h"
#include <set>
#include <fstream>


void NodeGraph::Load(const std::string& filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		std::cerr << "cannot open " << filename << std::endl;
		std::abort();
	}

	// load nodes
	int nodeSize, vertexSize, netDegree, knnSize;
	ifs >> nodeSize >> vertexSize >> netDegree >> knnSize;
	nodeIdx.resize(nodeSize);
	for (int i = 0; i < nodeSize; i++) 
		ifs >> nodeIdx[i];

	// load neighborCnt
	nodeNet.resize(netDegree, nodeSize);
	for (int i = 0; i < nodeSize; i++)
		for (int j = 0; j < netDegree; j++)
			ifs >> nodeNet(j, i);
	
	// load knn and weights
	knn.resize(knnSize, vertexSize);
	weight.resize(knnSize, vertexSize);
	for (int vi = 0; vi < vertexSize; vi++)
		for (int ki = 0; ki < knnSize; ki++)
			ifs >> knn(ki, vi) >> weight(ki, vi);
	
	ifs.close();
}


void NodeGraph::Save(const std::string& filename) const
{
	std::ofstream ofs(filename);

	ofs << nodeIdx.size() << "\t" << knn.cols() << "\t" << nodeNet.rows() << "\t" << knn.rows() << std::endl;
	for (int i = 0; i < nodeIdx.size(); i++)
		ofs << nodeIdx[i] << std::endl;
	ofs << std::endl;

	// load neighborCnt
	for (int i = 0; i < nodeIdx.size(); i++) {
		for (int j = 0; j < nodeNet.rows(); j++)
			ofs << nodeNet(j, i) << "\t";
		ofs << std::endl;
	}
	ofs << std::endl;

	// load knn and weights
	for (int vi = 0; vi < knn.cols(); vi++) {
		for (int ki = 0; ki < knn.rows(); ki++)
			ofs << knn(ki, vi) << "\t" << weight(ki, vi) << std::endl;
		ofs << std::endl;
	}

	ofs.close();
}


Eigen::VectorXf NodeGraphGenerator::Dijkstra(const int& start, const Eigen::MatrixXf& w) const
{
	const int n = (int)w.rows();
	Eigen::VectorXf dist = w.col(start);
	Eigen::VectorXi visited = Eigen::VectorXi::Zero(n);

	dist(start) = 0;
	visited(start) = 1;

	// add n+1 vertices
	for (int cnt = 1; cnt < n; cnt++) {
		// find the vertex unvisited nearest to start vertex
		int k = -1;
		float dmin = FLT_MAX;
		for (int i = 0; i < n; i++)
			if (!visited(i) && dist(i) < dmin) {
				dmin = dist(i);
				k = i;
			}

		if (k == -1) 
			break;

		visited(k) = 1;
		// update
		for (int i = 0; i < n; i++) {
			if (visited[i] || w(k, i) == FLT_MAX)
				continue;
			dist[i] = std::min(dist[i], dist(k) + w(k, i));
		}
	}
	return dist;
}


void NodeGraphGenerator::CalcGeodesic(const Model& model)
{
	const int n = (int)model.vertices.cols();
	geodesic.clear();
	geodesic.resize(n);

	// generate graph
	std::vector<std::map<int, float>> nbor(n);
	for (int fIdx = 0; fIdx < model.faces.cols(); fIdx++) {
		for (int i = 0; i < 3; i++) {
			const int va = model.faces(i, fIdx);
			const int vb = model.faces((i + 1) % 3, fIdx);
			const float norm = (model.vertices.col(va) - model.vertices.col(vb)).norm();
			nbor[va].insert(std::make_pair(vb, norm));
			nbor[vb].insert(std::make_pair(va, norm));
		}
	}

	// calc geodesic distance
	const float cutSpacing = cutRate * nodeSpacing;
#pragma omp parallel for
	for (int srcIdx = 0; srcIdx < n; srcIdx++) {
		std::vector<int> tarIdxMap;
		int start = -1;
		for (int tarIdx = 0; tarIdx < n; tarIdx++) {
			if (tarIdx == srcIdx)
				start = int(tarIdxMap.size());
			if ((model.vertices.col(tarIdx) - model.vertices.col(srcIdx)).norm() < cutSpacing)
				tarIdxMap.emplace_back(tarIdx);
		}

		Eigen::MatrixXf w = Eigen::MatrixXf::Constant(tarIdxMap.size(), tarIdxMap.size(), FLT_MAX);
		for (int i = 0; i < w.rows(); i++) {
			for (const auto& nb : nbor[tarIdxMap[i]]) {
				auto iter = std::find(tarIdxMap.begin(), tarIdxMap.end(), nb.first);
				if (iter != tarIdxMap.end())
					w(i, iter - tarIdxMap.begin()) = nb.second;
			}
		}

		Eigen::VectorXf dist = Dijkstra(start, w);

		for (int i = 0; i < tarIdxMap.size(); i++)
			if (dist[i] != FLT_MAX)
				geodesic[srcIdx].insert(std::make_pair(tarIdxMap[i], dist[i]));
	}
}


void NodeGraphGenerator::SampleNode()
{
	Eigen::VectorXi valid = Eigen::VectorXi::Ones(geodesic.size());

	for (int vIdx = 0; vIdx < geodesic.size(); vIdx++) 
		if (valid[vIdx]) 
			for(const auto& pair:geodesic[vIdx])
				if (valid(pair.first) && pair.first != vIdx && pair.second < nodeSpacing)
					valid[pair.first] = 0;

	nodeIdx.resize((valid.array() > 0).count());
	for (int i = 0, j = 0; i < geodesic.size(); i++)
		if (valid(i))
			nodeIdx[j++] = i;
}


void NodeGraphGenerator::GenKnn()
{
	knn = Eigen::MatrixXi::Constant(knnSize, geodesic.size(), -1);
	weight = Eigen::MatrixXf::Zero(knnSize, geodesic.size());

	for (int vIdx = 0; vIdx < geodesic.size(); vIdx++) {
		std::vector<std::pair<float, int>> dist;
		for (int ni = 0; ni < nodeIdx.size(); ni++) {
			if (nodeIdx[ni] == vIdx)
				continue;
			auto iter = geodesic[vIdx].find(nodeIdx[ni]);
			if (iter != geodesic[vIdx].end() && iter->second != FLT_MAX)
				dist.emplace_back(std::make_pair(iter->second, ni));
		}

		if (dist.size() < knnSize) 
			std::cerr << "dist size: "<< dist.size() << " geodesic is not enough, should turn up cut rate" << std::endl;

		std::sort(dist.begin(), dist.end());
		float sum = 0;
		for (int i = 0; i < dist.size() && i < knnSize; i++)
			sum+= 1.f / dist[i].first;

		for (int i = 0; i < dist.size() && i < knnSize; i++) {
			knn(i, vIdx) = dist[i].second;
			weight(i, vIdx) = 1.f / sum / dist[i].first;
		}
	}
}


void NodeGraphGenerator::GenNodeNet()
{
	nodeNet = Eigen::MatrixXi::Constant(netDegree, nodeIdx.size(), -1);
	for (int ni = 0; ni < nodeIdx.size(); ni++) {
		std::vector<std::pair<float, int>> dist;
		for (int nj = 0; nj < nodeIdx.size(); nj++)
			if (nj != ni) {
				auto iter = geodesic[nodeIdx[ni]].find(nodeIdx[nj]);
				if (iter != geodesic[nodeIdx[ni]].end() && iter->second != FLT_MAX)
					dist.emplace_back(std::make_pair(iter->second, nj));
			}

		if (dist.size() < netDegree)
			std::cerr << "dist size: " << dist.size() << " geodesic is not enough, should turn up cut rate" << std::endl;


		std::sort(dist.begin(), dist.end());
		for (int i = 0; i < dist.size() && i < netDegree; i++)
			nodeNet(i, ni) = dist[i].second;
	}
}


void NodeGraphGenerator::Generate(const Model& model)
{
	CalcGeodesic(model);
	SampleNode();
	GenKnn();
	GenNodeNet();
}


void NodeGraphGenerator::LoadGeodesic(const std::string& filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		std::cerr << "cannot open " << filename << std::endl;
		std::abort();
	}

	int m,n;
	std::pair<int, float> pair;
	ifs >> n;
	geodesic.resize(n);
	for (int i = 0; i < n; i++) {
		ifs >> m;
		for (int j = 0; j < m; j++) {
			ifs >> pair.first >> pair.second;
			geodesic[i].insert(pair);
		}
	}
	ifs.close();
}


void NodeGraphGenerator::SaveGeodesic(const std::string& filename) const
{
	std::ofstream ofs(filename);
	ofs << geodesic.size() << std::endl;
	for (const auto& geo : geodesic) {
		ofs << geo.size() << "\t";
		for (const auto& pair : geo)
			ofs << pair.first << "\t" << pair.second << "\t";
		ofs << std::endl;
	}
	ofs.close();
}


void NodeGraphGenerator::VisualizeNodeNet(const Model& model, const std::string& filename) const
{
	std::ofstream fs(filename);
	for (int i = 0; i < nodeIdx.size(); i++)
		fs << "v " << model.vertices(0, nodeIdx[i]) << " " << 
			model.vertices(1, nodeIdx[i]) << " " << model.vertices(2, nodeIdx[i]) << std::endl;

	for (int i = 0; i < nodeNet.cols(); i++)
		for (int j = 0; j < nodeNet.rows(); j++)
			if (nodeNet(j, i) != -1)
				fs << "l " << i + 1 << " " << nodeNet(j,i) + 1 << std::endl;
	fs.close();
}


void NodeGraphGenerator::VisualizeKnn(const Model& model, const std::string& filename) const
{
	std::ofstream fs(filename);
	for (int i = 0; i < knn.cols(); i++)
		fs << "v " << model.vertices(0, i) << " " << model.vertices(1, i) << " " << model.vertices(2, i) << std::endl;

	for (int i = 0; i < knn.cols(); i++)
		for (int j = 0; j < knn.rows(); j++)
			if (knn(j, i) != -1)
				fs << "l " << i + 1 << " " << nodeIdx[knn(j, i)] + 1 << std::endl;
	fs.close();
}
