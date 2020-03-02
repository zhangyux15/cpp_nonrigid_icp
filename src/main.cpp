#include "nonrigidICP.h"


int main()
{
	auto srcModel = std::make_shared<Model>("../data/src.obj");
	auto tarModel = std::make_shared<Model>("../data/tar.obj");
	NonrigidICP solver;
	solver.SetModel(srcModel, tarModel);
	solver.SetNodeGraph(std::make_shared<NodeGraph>("../data/graph.txt"));
	
	Eigen::VectorXf wDeform(5023);
	std::ifstream weightFile("../data/flame_weight.txt");
	for (int i = 0; i < wDeform.size(); i++)
		weightFile >> wDeform[i];
	weightFile.close();

	solver.SetWeight(wDeform, 1.f, 0.1f);
	solver.Solve(40, 1e-5f, "");
	solver.GetIterModel().Save("../data/tmp_1.obj");

	solver.SetWeight(wDeform, 0.1f, 0.1f);
	solver.SetCorrParam(0.05f, float(EIGEN_PI) / 4);
	solver.Solve(20, 1e-5f, "");
	solver.GetIterModel().Save("../data/tmp_2.obj");

	solver.SetWeight(wDeform, 0.01f, 0.01f);
	solver.SetCorrParam(0.01f, float(EIGEN_PI) / 4);
	solver.Solve(20, 1e-5f, "");

	solver.GetIterModel().Save("../data/final.obj");
	return 0;
}


//int main()
//{
//	NodeGraphGenerator generator;
//	Model model("../data/template.obj");
//	generator.Generate(model);
//	generator.Save("../data/graph.txt");
//	generator.VisualizeNodeNet("../data/nodeNet.obj");
//	generator.VisualizeKnn("../data/knn.obj");
//	system("pause");
//	return 0;
//}




//
//Model modelA("../data/template.obj");
//Model modelB("../data/ear.obj");
//for (int i = 0; i < modelB.vertices.cols(); i++) {
//	for (int j = 0; j < modelA.vertices.cols(); j++) {
//		if ((modelB.vertices.col(i) - modelA.vertices.col(i)).norm() < 1e-4f)
//		{
//			wDeform[j] = 0.f;
//			break;
//		}
//	}
//}
