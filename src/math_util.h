#pragma once
#include <cmath>
#include <fstream>
#include <type_traits>
#include <Eigen/Core>


namespace Eigen {
	typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
	typedef Eigen::Matrix<unsigned char, 3, 3> Matrix3b;
	typedef Eigen::Matrix<unsigned char, 3, Eigen::Dynamic> Matrix3Xb;
	typedef Eigen::Matrix<unsigned char, 4, Eigen::Dynamic> Matrix4Xb;
	typedef Eigen::Matrix<unsigned char, 2, 1> Vector2b;
	typedef Eigen::Matrix<unsigned char, 3, 1> Vector3b;
	typedef Eigen::Matrix<unsigned char, 4, 1> Vector4b;
	typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> MatrixXu;
	typedef Eigen::Matrix<unsigned int, 3, 3> Matrix3u;
	typedef Eigen::Matrix<unsigned int, 3, Eigen::Dynamic> Matrix3Xu;
	typedef Eigen::Matrix<unsigned int, 4, Eigen::Dynamic> Matrix4Xu;
	typedef Eigen::Matrix<unsigned int, 2, 1> Vector2u;
	typedef Eigen::Matrix<unsigned int, 3, 1> Vector3u;
	typedef Eigen::Matrix<unsigned int, 4, 1> Vector4u;
	typedef Eigen::Matrix<float, 6, 1> Vector6f;
	typedef Eigen::Matrix<float, 3, 4> Matrix34f;
	typedef Eigen::Matrix<float, 3, 2> Matrix32f;
	typedef Eigen::Matrix<double, 6, 1> Vector6d;
	typedef Eigen::Matrix<double, 3, 4> Matrix34d;
	typedef Eigen::Matrix<double, 3, 2> Matrix32d;
}


namespace MathUtil {
	// Linear Algebra
	inline Eigen::Matrix3f Skew(const Eigen::Vector3f& vec)
	{
		Eigen::Matrix3f skew;
		skew << 0, -vec.z(), vec.y(),
			vec.z(), 0, -vec.x(),
			-vec.y(), vec.x(), 0;
		return skew;
	}


	inline Eigen::Matrix4f Twist(const Eigen::Vector6f &_twist)
	{
		// calculate exponential mapping from Lie Algebra (se(3)) to Lie Group (SE(3))
		Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
		Eigen::Vector3f axis = _twist.head(3);

		if (axis.cwiseAbs().sum() > 1e-5f) {
			float angle = axis.norm();
			axis.normalize();

			// rotation
			T.topLeftCorner(3, 3) = Eigen::AngleAxisf(angle, axis).matrix();

			// translation
			Eigen::Vector3f rho(_twist.tail(3));
			const float s = std::sinf(angle) / angle;
			const float t = (1 - std::cosf(angle)) / angle;

			Eigen::Matrix3f skew = Skew(axis);
			Eigen::Matrix3f J = s * Eigen::Matrix3f::Identity() + (1 - s) * (skew * skew + Eigen::Matrix3f::Identity()) + t * skew;
			Eigen::Vector3f trans = J * rho;
			T.topRightCorner(3, 1) = trans;
		}
		return T;
	}
}

