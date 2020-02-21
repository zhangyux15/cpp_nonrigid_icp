#include "nonrigidICP.h"


//int main()
//{
//	auto srcModel = std::make_shared<Model>("../data/rigid_icp.obj");
//	auto tarModel = std::make_shared<Model>("../data/scan.obj");
//	auto graph = std::make_shared<NodeGraph>("../data/smplx_graph.txt");
//	NonrigidICP solver;
//	solver.SetModel(srcModel, tarModel);
//	solver.SetNodeGraph(graph);
//	solver.Solve(20, 1e-5f, "../data/");
//	return 0;
//}
//


int main()
{
	NodeGraphGenerator generator;
	Model model("../data/template.obj");
	generator.Generate(model);
	generator.Save("../data/graph.txt");
	generator.VisualizeNodeNet(model, "../data/nodeNet.obj");
	generator.VisualizeKnn(model, "../data/knn.obj");
	system("pause");
	return 0;
}