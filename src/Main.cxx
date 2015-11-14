/**
 * Christof Seiler
 * Stanford University, Department of Statistics
 */

#include <ctime>

#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkTimeProbe.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkBSplineTransform.h>
#include <itkGradientImageFilter.h>
#include <itkWarpImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkSquareImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkInverseDisplacementFieldImageFilter.h>
#include <itkIterativeInverseDisplacementFieldImageFilter.h>

//#include <random>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/program_options.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

const unsigned int Dimension = 3;
typedef itk::Image< signed short, Dimension > ImageType;
typedef float CoeffType;
typedef itk::Image< CoeffType, Dimension > CoefficientImageType;
typedef itk::Vector< CoeffType,Dimension > VectorType;
typedef itk::Image< VectorType,Dimension > VectorCoefficientImageType;

const CoeffType sparseRadius = 2;

using namespace Eigen;
namespace po = boost::program_options;

VectorCoefficientImageType::Pointer createDisplacementField(const MatrixXf& weights, CoefficientImageType::Pointer controlPointsImage, CoefficientImageType::Pointer fixedImage, const CoefficientImageType::RegionType& insideRegion) {
    
    const unsigned int totalControlPoints = insideRegion.GetNumberOfPixels();
    
    // resample moving image
    MatrixXf weightsX[Dimension];
    for(unsigned int i = 0; i < Dimension; ++i)
        weightsX[i] = weights.block(i*totalControlPoints,0,totalControlPoints,1);
    
    CoefficientImageType::Pointer weigthsVectorField[Dimension];
    typedef itk::BSplineInterpolateImageFunction< CoefficientImageType > BSplineInterpolatorType;
    BSplineInterpolatorType::Pointer interpolators[Dimension];
    for(unsigned int i = 0; i < Dimension; ++i) {
        weigthsVectorField[i] = CoefficientImageType::New();
        weigthsVectorField[i]->SetRegions(controlPointsImage->GetLargestPossibleRegion());
        weigthsVectorField[i]->SetSpacing(controlPointsImage->GetSpacing());
        weigthsVectorField[i]->Allocate();
        weigthsVectorField[i]->FillBuffer(0.0);
        
        //            std::cout << "weightsX[i].transpose() = \n" << weightsX[i].transpose() << std::endl;
        
        typedef itk::ImageRegionIteratorWithIndex< CoefficientImageType > IteratorTypeWithIndex;
        IteratorTypeWithIndex iterWeights(weigthsVectorField[i], weigthsVectorField[i]->GetLargestPossibleRegion());
        unsigned int controlPointIndex = 0;
        for(iterWeights.GoToBegin(); !iterWeights.IsAtEnd(); ++iterWeights) {
            if(insideRegion.IsInside(iterWeights.GetIndex())) {
                iterWeights.Set(weightsX[i](controlPointIndex,0));
                ++controlPointIndex;
            }
        }
        
        //            typedef itk::ImageFileWriter< CoefficientImageType > WriterControlType;
        //            WriterControlType::Pointer writerControlField = WriterControlType::New();
        //            writerControlField->SetFileName("WeigthsVectorField.mha");
        //            writerControlField->SetInput(weigthsVectorField[i]);
        //            writerControlField->Update();
        
        interpolators[i] = BSplineInterpolatorType::New();
        interpolators[i]->SetInputImage(weigthsVectorField[i]);
    }
    
    VectorCoefficientImageType::Pointer displacementField = VectorCoefficientImageType::New();
    displacementField->CopyInformation(fixedImage);
    displacementField->SetRegions(fixedImage->GetLargestPossibleRegion());
    displacementField->SetSpacing(fixedImage->GetSpacing());
    displacementField->Allocate();
    
    typedef itk::ImageRegionIteratorWithIndex< VectorCoefficientImageType > IteratorTypeWithIndex;
    IteratorTypeWithIndex iterField(displacementField, displacementField->GetLargestPossibleRegion());
    for(iterField.GoToBegin(); !iterField.IsAtEnd(); ++iterField) {
        VectorCoefficientImageType::PointType point;
        displacementField->TransformIndexToPhysicalPoint(iterField.GetIndex(), point);
        
        VectorType value;
        for(unsigned int i = 0; i < Dimension; ++i)
            value[i] = interpolators[i]->Evaluate(point);
        iterField.Set(value);
    }
    
    return displacementField;

}

MatrixXf imageToMatrix(CoefficientImageType::Pointer image) {
    MatrixXf matrix(image->GetLargestPossibleRegion().GetNumberOfPixels(),1);
    typedef itk::ImageRegionConstIterator< CoefficientImageType > CoeffIteratorType;
    CoeffIteratorType iterCoeff(image, image->GetLargestPossibleRegion());
    unsigned int pixelIndex = 0;
    for(iterCoeff.GoToBegin(); !iterCoeff.IsAtEnd(); ++iterCoeff) {
        matrix(pixelIndex,0) = iterCoeff.Get();
        ++pixelIndex;
    }
    return matrix;
}

void saveMatrixToFile(const MatrixXf& matrix, const char* fileName) {
    std::ofstream matrixFile;
    matrixFile.open (fileName);
    matrixFile << matrix;
    matrixFile.close();
}

MatrixXf pinv( const MatrixXf& pinvmat)
{
    JacobiSVD<MatrixXf> svd(pinvmat, ComputeFullU | ComputeFullV);
    VectorXf m_singularValues = svd.singularValues();
    MatrixXf m_matrixV = svd.matrixV();
    MatrixXf m_matrixU = svd.matrixU();
    
    double pinvtoler = 1e-15*m_singularValues.maxCoeff();
    VectorXf singularValues_inv=m_singularValues;
    for ( long i=0; i<pinvmat.cols(); ++i) {
        if ( m_singularValues(i) > pinvtoler )
            singularValues_inv(i)=1.0/m_singularValues(i);
        else singularValues_inv(i)=0;
    }
    return m_matrixV*singularValues_inv.asDiagonal()*m_matrixU.transpose();
}

void computeJ(const SparseMatrix<CoeffType>& Bmatrix, CoefficientImageType::Pointer image, SparseMatrix<CoeffType>& J) {
    
    // compute gradient of resampled moving image
    typedef itk::GradientImageFilter< CoefficientImageType > GradientImageFilterType;
    GradientImageFilterType::Pointer gradientFilter = GradientImageFilterType::New();
    gradientFilter->SetInput(image);
    gradientFilter->Update();

    typedef itk::VectorIndexSelectionCastImageFilter< GradientImageFilterType::OutputImageType,CoefficientImageType > IndexSelectionType;
    MatrixXf gradMVec[Dimension];
//    #pragma omp parallel for
    for(unsigned int i = 0; i < Dimension; ++i) {
        IndexSelectionType::Pointer indexSelectionFilter = IndexSelectionType::New();
        indexSelectionFilter->SetIndex(i);
        indexSelectionFilter->SetInput(gradientFilter->GetOutput());
        indexSelectionFilter->Update();
        gradMVec[i] = imageToMatrix(indexSelectionFilter->GetOutput());
        
        //            // test
        //            std::ostringstream oss;
        //            oss << "TestGradient" << i << ".mha";
        //            writer->SetInput(indexSelectionFilter->GetOutput());
        //            writer->SetFileName(oss.str());
        //            writer->Update();
    }

    // gradient of cost function wrt spline parameters
    //    MatrixXf Jx[Dimension];
    
    if(J.nonZeros() == 0) {
//    if(true) {
    
        itk::TimeProbe clockFillList;
        clockFillList.Start();
        
        typedef Eigen::Triplet<CoeffType> T;
        std::vector<T> tripletList;
        tripletList.reserve(Dimension*Bmatrix.nonZeros());
        
        //    MatrixXf Jx[3];
        for(unsigned int i = 0; i < Dimension; ++i) {
            //        Jx[i] = MatrixXf::Zero(totalNumberOfPixels,totalControlPoints);
            for(unsigned int j = 0; j < Bmatrix.cols(); ++j) {
                SparseVector<CoeffType> vec = Bmatrix.col(j).cwiseProduct(gradMVec[i]);
                for (SparseVector<CoeffType>::InnerIterator it(vec); it; ++it)
                    tripletList.push_back( T(it.index(),(i*Bmatrix.cols())+j,it.value()) );
            }
            //            Jx[i].col(j) = gradMVec[i].cwiseProduct(Bmatrix.col(j));
        }
        
        clockFillList.Stop();
        std::cout << "push back triplets: " << clockFillList.GetTotal() << std::endl;

        itk::TimeProbe clockCreateJ;
        clockCreateJ.Start();
        
        J.setFromTriplets(tripletList.begin(), tripletList.end());
        
        clockCreateJ.Stop();
        std::cout << "create J from triplets: " << clockCreateJ.GetTotal() << std::endl;
    }
    else {
        itk::TimeProbe clockUpdateJ;
        clockUpdateJ.Start();
        
        for(unsigned int i = 0; i < Dimension; ++i) {
            #pragma omp parallel for
            for(unsigned int j = 0; j < Bmatrix.cols(); ++j) {
                SparseVector<CoeffType> vec = Bmatrix.col(j).cwiseProduct(gradMVec[i]);
                for (SparseVector<CoeffType>::InnerIterator it(vec); it; ++it)
                    J.coeffRef(it.index(), (i*Bmatrix.cols())+j) = it.value();
            }
        }
        
        clockUpdateJ.Stop();
        std::cout << "update J coefficient-wise: " << clockUpdateJ.GetTotal() << std::endl;
    }

    //    // test
    //    saveMatrixToFile(Jx[0], "Jx0.txt");
    //    saveMatrixToFile(Jx[1], "Jx1.txt");
    //    saveMatrixToFile(Bmatrix, "Bmatrix.txt");
    //    saveMatrixToFile(gradMVec[0], "gradMVec0.txt");
    //    saveMatrixToFile(gradMVec[1], "gradMVec1.txt");

    //    MatrixXf J = MatrixXf::Zero(J.rows(),J.cols());
    //    for(unsigned int i = 0; i < Dimension; ++i)
    //        J.block(0,i*totalControlPoints,totalNumberOfPixels,totalControlPoints) = Jx[i];

    //    // test
    //    saveMatrixToFile(J, "J.txt");
}

MatrixXf gradV(const float alpha, const float lam, const SparseMatrix<CoeffType>& Bmatrix, const SparseMatrix<CoeffType>& S, const MatrixXf& weights, CoefficientImageType::Pointer controlPointsImage, CoefficientImageType::Pointer fixedImage, CoefficientImageType::Pointer movingImage, const CoefficientImageType::RegionType& insideRegion, float& ssd, SparseMatrix<CoeffType>& J) {
    
    // resample moving image
    itk::TimeProbe clockDisp;
    clockDisp.Start();
    VectorCoefficientImageType::Pointer displacementField = createDisplacementField(weights, controlPointsImage, fixedImage, insideRegion);
    clockDisp.Stop();
    std::cout << "create displacement field: " << clockDisp.GetTotal() << std::endl;
    
    typedef itk::WarpImageFilter< CoefficientImageType,CoefficientImageType,VectorCoefficientImageType > WrapImageFilterType;
    WrapImageFilterType::Pointer wrapFilter = WrapImageFilterType::New();
    wrapFilter->SetOutputSpacing(displacementField->GetSpacing());
    wrapFilter->SetOutputDirection(displacementField->GetDirection());
    wrapFilter->SetOutputOrigin(displacementField->GetOrigin());
    wrapFilter->SetOutputSize(displacementField->GetLargestPossibleRegion().GetSize());
    wrapFilter->SetInput(movingImage);
    wrapFilter->SetDisplacementField(displacementField);
    wrapFilter->Update();
    
    // compute residual error
    typedef itk::SubtractImageFilter< CoefficientImageType > SubtractImageFilterType;
    SubtractImageFilterType::Pointer subtractFilter = SubtractImageFilterType::New();
    subtractFilter->SetInput1(wrapFilter->GetOutput());
    subtractFilter->SetInput2(fixedImage);
    subtractFilter->Update();
    
    // compute sum of squared differences
    typedef itk::SquareImageFilter< CoefficientImageType,CoefficientImageType > SquareImageFilterType;
    SquareImageFilterType::Pointer squareFilter = SquareImageFilterType::New();
    squareFilter->SetInput(subtractFilter->GetOutput());
    typedef itk::StatisticsImageFilter< CoefficientImageType > StatisticsImageFilterType;
    StatisticsImageFilterType::Pointer statisticsFilter = StatisticsImageFilterType::New();
    statisticsFilter->SetInput(squareFilter->GetOutput());
    statisticsFilter->Update();
    ssd = statisticsFilter->GetSum();
    std::cout << "ssd = " << ssd << std::endl;
    
    VectorXf residualErrorVec = imageToMatrix(subtractFilter->GetOutput()).col(0);
//    // test
//    saveMatrixToFile(residualErrorVec, "residualErrorVec.txt");
    
    computeJ(Bmatrix, wrapFilter->GetOutput(), J);
    
    itk::TimeProbe clockGrad;
    clockGrad.Start();

//    MatrixXf gradCw = alpha*SparseMatrix<CoeffType>(J.transpose())*residualErrorVec + lam*S*weights;
    // faster: columnwise multiplication with parallel for loop
    MatrixXf gradCw(weights.rows(),weights.cols());
    #pragma omp parallel for
    for(unsigned int i = 0; i < J.cols(); ++i) {
        VectorXf Jvec = J.col(i);
        gradCw(i,0) = Jvec.dot(residualErrorVec);
    }
    gradCw = alpha*gradCw + lam*S*weights;
    
    clockGrad.Stop();
    std::cout << "compute gradCw: " << clockGrad.GetTotal() << std::endl;
    
    return gradCw;
}

SparseMatrix<CoeffType> hessV(const float alpha, const float lam, SparseMatrix<CoeffType>& J, const SparseMatrix<CoeffType>& S) {
    itk::TimeProbe clockHess;
    clockHess.Start();

    SparseMatrix<CoeffType> HessianCw = alpha*SparseMatrix<CoeffType>(J.transpose())*J + lam*S;
    
    clockHess.Stop();
    std::cout << "compute HessianCw: " << clockHess.GetTotal() << std::endl;

    return HessianCw;
}

float potentialV(const MatrixXf& q, const float alpha, const float lam, const SparseMatrix<CoeffType>& S, const float ssd) {
    return alpha*ssd + (lam*q.transpose()*S*q)(0,0);
}

void saveResults(CoefficientImageType::Pointer movingImage, VectorCoefficientImageType::Pointer displacementField,  unsigned int step, const char* postfix) {
    std::ostringstream ossField;
    ossField << "DisplacementField_" << postfix << "_Step_" << step << ".mha";
    typedef itk::ImageFileWriter< VectorCoefficientImageType > WriterFieldType;
    WriterFieldType::Pointer writerField = WriterFieldType::New();
    writerField->SetFileName(ossField.str());
    writerField->SetInput(displacementField);
    writerField->UseCompressionOn();
    writerField->Update();
    
    typedef itk::WarpImageFilter< CoefficientImageType,CoefficientImageType,VectorCoefficientImageType > WrapImageFilterType;
    WrapImageFilterType::Pointer wrapFilter = WrapImageFilterType::New();
    wrapFilter->SetOutputSpacing(displacementField->GetSpacing());
    wrapFilter->SetOutputDirection(displacementField->GetDirection());
    wrapFilter->SetOutputOrigin(displacementField->GetOrigin());
    wrapFilter->SetOutputSize(displacementField->GetLargestPossibleRegion().GetSize());
    wrapFilter->SetInput(movingImage);
    wrapFilter->SetDisplacementField(displacementField);
    wrapFilter->Update();
    
    std::ostringstream ossWarpedMoving;
    ossWarpedMoving << "WarpedMoving_" << postfix << "_Step_" << step << ".mha";
    typedef itk::ImageFileWriter< CoefficientImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput(wrapFilter->GetOutput());
    writer->SetFileName(ossWarpedMoving.str());
    writer->UseCompressionOn();
    writer->Update();
}

int main( int argc, char** argv ) {
    
    std::cout << "number of thread used by Eigen: " << Eigen::nbThreads() << std::endl;
    std::cout << "Dimension = " << Dimension << std::endl;

    std::string fixedImageName;
    std::string movingImageName;
    std::string outputName;
    CoefficientImageType::SizeType noOfControlPoints;
    bool verboseResults = false;
    
    unsigned int noOfSteps = 100;
    float alpha = 1.0;
    float lam = 0.1;
    float Kstd = 1.0;
    float epsilon = 0.1;
    float L = 10;
    unsigned int T = 500;
    
    bool warpMode = false;
    std::string applyMethod = "Undefined";
    unsigned int applyStep = 0;
    bool invertDVF = false;
    bool hessianProposal = false;
    
    unsigned int pickControlPoint = 0;
    
    try {
        // Declare the supported options.
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("fixed", po::value<std::string>()->required(), "set path to fixed image")
            ("moving", po::value<std::string>()->required(), "set path to moving image")
            ("output-prefix", po::value<std::string>()->required(), "set prefix for all output files")
            ("cx", po::value<unsigned int>()->required(), "set number of control point in x direction")
            ("cy", po::value<unsigned int>()->required(), "set number of control point in y direction")
            ("cz", po::value<unsigned int>()->required(), "set number of control point in z direction")
            ("verbose","save intermediate matrices and images to file")
            ("noOfSteps", po::value<unsigned int>(), "set number of Gauss-Newton steps")
            ("lam", po::value<float>(), "set prior importance: likelihood weight = 1.0, prior weight = lam")
            ("Kstd", po::value<float>(), "set standard deviation of normal for momenta")
            ("epsilon", po::value<float>(), "set leapfrog stepsize")
            ("L", po::value<float>(), "set leapfrog integration steps")
            ("T", po::value<float>(), "set number of HMC steps")
            ("warpMode","switch to warp mode")
            ("applyMethod", po::value<std::string>(), "set method name for displacement field and warp output files")
            ("applyStep", po::value<unsigned int>(), "set step for displacement field and warp output files")
            ("invertDVF","switch to invert DVF for warping")
            ("hessianProposal","set to use inverse of the Hessian as the proposal")
            ("pickControlPoint", po::value<unsigned int>(), "set control point for visualization")
        ;
        
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if(vm.count("help")) {
            std::cout << desc << "\n";
            return EXIT_SUCCESS;
        }
        po::notify(vm);
        
        fixedImageName = vm["fixed"].as<std::string>();
        movingImageName = vm["moving"].as<std::string>();
        outputName = vm["output-prefix"].as<std::string>();
        
        noOfControlPoints[0] = vm["cx"].as<unsigned int>();
        noOfControlPoints[1] = vm["cy"].as<unsigned int>();
        noOfControlPoints[2] = vm["cz"].as<unsigned int>();

        if(vm.count("verbose")) verboseResults = true;
        if(vm.count("noOfSteps")) noOfSteps = vm["noOfSteps"].as<unsigned int>();
        
        if(vm.count("lam")) lam = vm["lam"].as<float>();
        if(vm.count("Kstd")) Kstd = vm["Kstd"].as<float>();
        if(vm.count("epsilon")) epsilon = vm["epsilon"].as<float>();
        if(vm.count("L")) L = vm["L"].as<float>();
        if(vm.count("T")) T = vm["T"].as<float>();
        
        if(vm.count("warpMode")) warpMode = true;
        if(vm.count("applyMethod")) applyMethod = vm["applyMethod"].as<std::string>();
        if(vm.count("applyStep")) applyStep = vm["applyStep"].as<unsigned int>();
        if(vm.count("invertDVF")) invertDVF = true;
        if(vm.count("hessianProposal")) hessianProposal = true;
        
        if(vm.count("pickControlPoint")) pickControlPoint = vm["pickControlPoint"].as<unsigned int>();
        
        std::cout << "--------------------------------- input parameters -------------------------------\n";
        std::cout << "fixed                             " << fixedImageName << "\n";
        std::cout << "moving                            " << movingImageName << "\n";
        std::cout << "output-prefix                     " << outputName << "\n";
        std::cout << "number of control points          ";
        for(unsigned int i = 0; i < Dimension; ++i) std::cout << noOfControlPoints[i] << " ";
        std::cout << "\n";
        std::cout << "verbose                           " << (verboseResults ? "true" : "false") << "\n";
        std::cout << "number of Gauss-Newton steps      " << noOfSteps << "\n";
        std::cout << "prior weight                      " << lam << "\n";
        std::cout << "standard devitation for momenta   " << Kstd << "\n";
        std::cout << "leapfrog stepsize                 " << epsilon << "\n";
        std::cout << "leapfrog integration steps        " << L << "\n";
        std::cout << "number of HMC steps               " << T << "\n";
        std::cout << "warp mode                         " << (warpMode ? "true" : "false") << "\n";
        if(warpMode) {
            std::cout << "method name for output files      " << applyMethod << "\n";
            std::cout << "step for output files             " << applyStep << "\n";
            std::cout << "invert dvf for warping            " << (invertDVF ? "true" : "false") << "\n";
        }
        std::cout << "inverse Hessian as proposal mode  " << (hessianProposal ? "true" : "false") << "\n";
        std::cout << "control point for visualization   " << pickControlPoint << "\n";
        std::cout << "----------------------------------------------------------------------------------\n";
    }
    catch(std::exception& e) {
        std::cerr << "command line parsing error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }
    
    itk::TimeProbe clock;
    clock.Start();
    
    // read fixed and moving images
    
    typedef itk::ImageFileReader< ImageType > ImageReaderType;
    typedef itk::CastImageFilter< ImageType, CoefficientImageType > CastFilterType;
    typedef itk::RescaleIntensityImageFilter< CoefficientImageType > RescaleFilterType;
    
    ImageReaderType::Pointer reader = ImageReaderType::New();
    reader->SetFileName(fixedImageName);
    CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput(reader->GetOutput());
    RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(castFilter->GetOutput());
    rescaleFilter->SetOutputMinimum(0.0);
    rescaleFilter->SetOutputMaximum(1.0);
    rescaleFilter->Update();

    CoefficientImageType::Pointer fixedImage = CoefficientImageType::New();
    if(warpMode)
        fixedImage = castFilter->GetOutput();
    else
        fixedImage = rescaleFilter->GetOutput();
    fixedImage->DisconnectPipeline();
    
    reader->SetFileName(movingImageName);
    castFilter->SetInput(reader->GetOutput());
    rescaleFilter->SetInput(castFilter->GetOutput());
    rescaleFilter->Update();
    
    CoefficientImageType::Pointer movingImage = CoefficientImageType::New();
    if(warpMode)
        movingImage = castFilter->GetOutput();
    else
        movingImage = rescaleFilter->GetOutput();
    movingImage->DisconnectPipeline();
    
    if(verboseResults) {
        typedef itk::ImageFileWriter< CoefficientImageType > WriterType;
        WriterType::Pointer writer = WriterType::New();
        writer->SetInput(fixedImage);
        writer->SetFileName("FixedImage.mha");
        writer->Update();
        writer->SetInput(movingImage);
        writer->SetFileName("MovingImage.mha");
        writer->Update();
    }
    
    // checks if fixed and moving are equivalent
    
    std::cout << "original fixedImage->GetOrigin(): " << fixedImage->GetOrigin() << std::endl;
    std::cout << "original movingImage->GetOrigin(): " << movingImage->GetOrigin() << std::endl;
    
    std::cout << "original fixedImage->GetSpacing(): " << fixedImage->GetSpacing() << std::endl;
    std::cout << "original movingImage->GetSpacing(): " << movingImage->GetSpacing() << std::endl;
    
    std::cout << "original fixedImage->GetLargestPossibleRegion(): " << fixedImage->GetLargestPossibleRegion() << std::endl;
    std::cout << "original movingImage->GetLargestPossibleRegion(): " << movingImage->GetLargestPossibleRegion() << std::endl;
    
    std::cout << "original fixedImage->GetDirection(): " << fixedImage->GetDirection() << std::endl;
    std::cout << "original movingImage->GetDirection(): " << movingImage->GetDirection() << std::endl;
    
    // create spline coefficients matrices
    
    std::cout << "noOfControlPoints = " << noOfControlPoints << std::endl;
    
    ImageType::RegionType region = fixedImage->GetLargestPossibleRegion();
    ImageType::SizeType size = region.GetSize();
    std::cout << "size: " << size << std::endl;
    ImageType::SpacingType spacing = fixedImage->GetSpacing();
    std::cout << "spacing: " << spacing << std::endl;
    ImageType::SpacingType physicalSize;
    for(unsigned int i = 0; i < Dimension; ++i)
        physicalSize[i] = size[i]*spacing[i];
    std::cout << "physicalSize: " << physicalSize << std::endl;
    
    CoefficientImageType::SpacingType coeffSpacing;
    for(unsigned int i = 0; i < Dimension; ++i)
        coeffSpacing[i] = ((double)physicalSize[i])/(noOfControlPoints[i]-1);
    std::cout << "coeffSpacing: " << coeffSpacing << std::endl;
    
    CoefficientImageType::RegionType coeffRegion = region;
    coeffRegion.SetSize(noOfControlPoints);
    
    CoefficientImageType::Pointer controlPointsImage = CoefficientImageType::New();
    controlPointsImage->SetRegions(coeffRegion);
    controlPointsImage->SetSpacing(coeffSpacing);
    controlPointsImage->Allocate();
    
    // inside region with non-zero control points
    CoefficientImageType::IndexType insideIndex;
    CoefficientImageType::SizeType insideSize;
    for(unsigned int i = 0; i < Dimension; ++i) {
        insideIndex[i] = 1;
        insideSize[i] = noOfControlPoints[i]-2;
    }
    CoefficientImageType::RegionType insideRegion(insideIndex,insideSize);
    std::cout << "insideRegion: " << insideRegion << std::endl;
    
    const unsigned int totalControlPoints = insideRegion.GetNumberOfPixels();
    const unsigned int totalNumberOfPixels = fixedImage->GetLargestPossibleRegion().GetNumberOfPixels();
    std::cout << "totalControlPoints: " << totalControlPoints << std::endl;
    std::cout << "totalNumberOfPixels: " << totalNumberOfPixels << std::endl;
//    MatrixXf Bmatrix = MatrixXf::Zero(totalNumberOfPixels,totalControlPoints);
//    MatrixXf Bx[Dimension];
//    for(unsigned int i = 0; i < Dimension; ++i)
//        Bx[i] = MatrixXf::Zero(totalNumberOfPixels,totalControlPoints);
    
    // compute displacement field and warp moving image
    if(warpMode) {
        std::cout << "read eigen matrix from file" << std::endl;
        
        std::ifstream weightsStream(outputName.c_str());
        std::string line;
        MatrixXf weights = Eigen::MatrixXf::Zero(Dimension*totalControlPoints, 1);
        unsigned int lineNumber = 0;
        while (getline(weightsStream, line)) {
            std::istringstream ss(line);
//            for(unsigned int i = 0; i < weights.size(); ++i)
            ss >> weights(lineNumber,0);
            ++lineNumber;
        }
        weightsStream.close();
        std::cout << "Number of lines read = " << lineNumber << std::endl;
        
        std::cout << "compute displacement field" << std::endl;
        VectorCoefficientImageType::Pointer displacementField = createDisplacementField(weights, controlPointsImage, fixedImage, insideRegion);
        
        // inverte displacement vector
        if(invertDVF) {
            typedef itk::IterativeInverseDisplacementFieldImageFilter< VectorCoefficientImageType,VectorCoefficientImageType > InvertDVFFilterType;
            InvertDVFFilterType::Pointer invertDVFFilter = InvertDVFFilterType::New();
            invertDVFFilter->SetInput(displacementField);
            invertDVFFilter->Update();
            displacementField = invertDVFFilter->GetOutput();
            displacementField->DisconnectPipeline();
        }
        
        std::cout << "warp moving image with spline weights computing using method: " << applyMethod << std::endl;
        std::cout << "at step: " << applyStep << std::endl;
        
        saveResults(movingImage, displacementField, applyStep, applyMethod.c_str());
        
        return EXIT_SUCCESS;
    }
    
    typedef itk::ImageRegionIteratorWithIndex< CoefficientImageType > IteratorTypeWithIndex;
    IteratorTypeWithIndex iterContolPoints(controlPointsImage, controlPointsImage->GetLargestPossibleRegion());
    unsigned int controlPointIndex = 0;
    
    itk::Vector<CoeffType,Dimension> meshSpacing = coeffSpacing;
    const CoeffType nonZeroRadius = sparseRadius*meshSpacing.GetNorm();
    std::cout << "nonZeroRadius = " << nonZeroRadius << std::endl;

    typedef Eigen::Triplet<double> TripletType;
    std::vector<TripletType> tripletList;
    std::vector<TripletType> tripletListBx[Dimension];
    unsigned int volumeSparseSphere = 4.0/3.0*itk::Math::pi*std::pow(nonZeroRadius,Dimension);
    std::cout << "volumeSparseSphere = " << volumeSparseSphere << std::endl;
    float spacingFactor = 1;
    for(unsigned int i = 0; i < Dimension; ++i)
        spacingFactor *= spacing[i];
    unsigned int noOfNonZeroVoxels = volumeSparseSphere*totalControlPoints/spacingFactor;
    std::cout << "noOfNonZeroVoxels = " << noOfNonZeroVoxels << std::endl;
    tripletList.reserve(noOfNonZeroVoxels);
    for(unsigned int i = 0; i < Dimension; ++i)
        tripletListBx[i].reserve(noOfNonZeroVoxels);
    
    std::cout << "creating spline coefficients for control point: ";
    for(iterContolPoints.GoToBegin(); !iterContolPoints.IsAtEnd(); ++iterContolPoints) {
        
        // test
//        CoefficientImageType::Pointer testbImage = CoefficientImageType::New();
//        testbImage->SetRegions(fixedImage->GetLargestPossibleRegion());
//        testbImage->CopyInformation(fixedImage);
//        testbImage->Allocate();
        
        std::cout << controlPointIndex << " " << std::flush;
        controlPointsImage->FillBuffer(0.0);
        if(insideRegion.IsInside(iterContolPoints.GetIndex())) {
//            std::cout << "(inside) " << std::flush;
            iterContolPoints.Set(1.0);
            
            CoefficientImageType::PointType controlPoint;
            controlPointsImage->TransformIndexToPhysicalPoint(iterContolPoints.GetIndex(), controlPoint);
    
            typedef itk::BSplineInterpolateImageFunction< CoefficientImageType > BSplineInterpolatorType;
            BSplineInterpolatorType::Pointer interpolator = BSplineInterpolatorType::New();
            interpolator->SetInputImage(controlPointsImage);
        
            IteratorTypeWithIndex iterImage(fixedImage, fixedImage->GetLargestPossibleRegion());
            unsigned int pixelIndex = 0;
            for(iterImage.GoToBegin(); !iterImage.IsAtEnd(); ++iterImage) {
                ImageType::PointType point;
                fixedImage->TransformIndexToPhysicalPoint(iterImage.GetIndex(), point);
                
                itk::Vector<CoeffType,Dimension> difference = controlPoint - point;
                if(difference.GetNorm() < nonZeroRadius) {

                    BSplineInterpolatorType::OutputType value;
                    BSplineInterpolatorType::CovariantVectorType deriv;
                    interpolator->EvaluateValueAndDerivative(point, value, deriv);
                    
                    // test
        //            testbImage->SetPixel(iterImage.GetIndex(), value);
                    
//                    Bmatrix(pixelIndex,controlPointIndex) = value;
                    tripletList.push_back(TripletType(pixelIndex,controlPointIndex,value));
                    
                    for(unsigned int i = 0; i < Dimension; ++i)
                        tripletListBx[i].push_back(TripletType(pixelIndex,controlPointIndex,deriv[i]));
//                        Bx[i](pixelIndex,controlPointIndex) = deriv[i];
                }
                ++pixelIndex;
            }
            ++controlPointIndex;
        }
        
//        // test
//        writer->SetInput(testbImage);
//        writer->SetFileName("testbImage.mha");
//        writer->Update();
        
    }
    
    SparseMatrix<CoeffType> Bmatrix(totalNumberOfPixels,totalControlPoints);
    Bmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
//    std::ofstream sparseMatrixFile;
//    sparseMatrixFile.open ("BmatrixSparse.txt");
//    sparseMatrixFile << Bmatrix;
//    sparseMatrixFile.close();
    
    std::cout << std::endl;
    std::cout << "Bmatrix number of nonzeros: " << Bmatrix.nonZeros() << std::endl;
    
    SparseMatrix<CoeffType> Bx[Dimension];
    for(unsigned int i = 0; i < Dimension; ++i) {
        Bx[i].resize(totalNumberOfPixels,totalControlPoints);
        Bx[i].setFromTriplets(tripletListBx[i].begin(), tripletListBx[i].end());
    }
    
    std::cout << std::endl;
    clock.Stop();
    
//    // test B matrices
//    saveMatrixToFile(Bmatrix, "Bmatrix.txt");
//    saveMatrixToFile(Bx[0], "Bx.txt");
//    saveMatrixToFile(Bx[1], "By.txt");
    
    if(verboseResults) {
        std::cout << "pickControlPoint: " << pickControlPoint << std::endl;
        MatrixXf bvectors[Dimension+1];
        bvectors[0] = Bmatrix.col(pickControlPoint);
        //std::cout << "bvectors[0]:\n" << bvectors[0] << std::endl;
        for(unsigned int i = 0; i < Dimension; ++i)
            bvectors[i+1] = Bx[i].col(pickControlPoint);

        std::vector<std::string> fileNames(4);
        fileNames[0] = "CoeffImage.mha";
        for(unsigned int i = 0; i < Dimension; ++i) {
            std::ostringstream ossDeriv;
            ossDeriv << "Deriv" << i << "Image.mha";
            fileNames[i+1] = ossDeriv.str();
        }
        
        for(unsigned int i = 0; i < Dimension+1; ++i) {
            CoefficientImageType::Pointer bImage = CoefficientImageType::New();
            bImage->SetRegions(fixedImage->GetLargestPossibleRegion());
            bImage->CopyInformation(fixedImage);
            bImage->Allocate();
            
            unsigned int pixelIndex = 0;
            typedef itk::ImageRegionIterator< CoefficientImageType > IteratorType;
            IteratorType iterBImage(bImage, bImage->GetLargestPossibleRegion());
            for(iterBImage.GoToBegin(); !iterBImage.IsAtEnd(); ++iterBImage) {
                iterBImage.Set( bvectors[i](pixelIndex) );
                ++pixelIndex;
            }
            
            typedef itk::ImageFileWriter< CoefficientImageType > WriterType;
            WriterType::Pointer writer = WriterType::New();
            writer->SetFileName(fileNames[i]);
            writer->SetInput(bImage);
            writer->Update();
        }
    }
    
    clock.Start();
    
    // regularization: membrane energy
    
//    MatrixXf Sxyz = MatrixXf::Zero(totalControlPoints,totalControlPoints);
    SparseMatrix<CoeffType> Sxyz(totalControlPoints,totalControlPoints);
    for(unsigned int i = 0; i < Dimension; ++i) {
        Sxyz += SparseMatrix<CoeffType>(Bx[i].transpose())*Bx[i];
    }
//    MatrixXf S = MatrixXf::Zero(Dimension*totalControlPoints,Dimension*totalControlPoints);
//    for(unsigned int i = 0; i < Dimension; ++i)
//        S.block(i*totalControlPoints,i*totalControlPoints,totalControlPoints,totalControlPoints) = Sxyz;
    std::vector<TripletType> tripletListS;
    tripletListS.reserve(Dimension*Sxyz.nonZeros());
    for (int k=0; k<Sxyz.outerSize(); ++k) {
        for (SparseMatrix<CoeffType>::InnerIterator it(Sxyz,k); it; ++it) {
            for(unsigned int i = 0; i < Dimension; ++i)
                tripletListS.push_back( TripletType((i*totalControlPoints)+it.row(),(i*totalControlPoints)+it.col(),it.value()) );
        }
    }
    SparseMatrix<CoeffType> S(Dimension*totalControlPoints,Dimension*totalControlPoints);
    S.setFromTriplets(tripletListS.begin(), tripletListS.end());

//    saveMatrixToFile(S, "S.txt");
    
    clock.Stop();
    
//    // Gauss-Newton optimization
//    clock.Start();
//    
////    typedef itk::BSplineTransform< CoeffType,Dimension > TransformType;
////    TransformType::Pointer transform = TransformType::New();
////    transform->SetTransformDomainOrigin( origin );
////    transform->SetTransformDomainPhysicalDimensions(physicalSize);
////    transform->SetTransformDomainDirection( direction );
////    transform->SetTransformDomainMeshSize(noOfControlPoints);
//    
//    // initalize paramters
    MatrixXf weights = Eigen::MatrixXf::Zero(Dimension*totalControlPoints, 1);
    MatrixXf weightsAll = Eigen::MatrixXf::Zero(Dimension*totalControlPoints, noOfSteps);
    std::cout << "alpha = " << alpha << ", lam = " << lam << std::endl;
    VectorXf ssds(noOfSteps);
//
////    // test createDisplacementField
////    MatrixXf weightsX(totalControlPoints,1);
////    weightsX.fill(5.0);
////    MatrixXf weightsY(totalControlPoints,1);
////    weightsY.fill(10.0);
////    weights.block(0,0,totalControlPoints,1) = weightsX;
////    weights.block(totalControlPoints,0,totalControlPoints,1) = weightsY;
////    VectorCoefficientImageType::Pointer testDisplacementField = createDisplacementField(weights, controlPointsImage, fixedImage, insideRegion);
//    
//    // precompute the inverse of HessianCW
    SparseMatrix<CoeffType> Jfixed(totalNumberOfPixels,Dimension*totalControlPoints);
//    computeJ(Bmatrix, fixedImage, Jfixed);
//    SparseMatrix<CoeffType> HessianCw = hessV(alpha, lam, Jfixed, S);
//    
//    itk::TimeProbe clockInverse;
//    clockInverse.Start();
//    
//    ConjugateGradient< SparseMatrix<CoeffType> > solver; // use ConjugateGradient for large sparse matrices / SimplicialLLT
//    solver.compute(HessianCw);
//    if(solver.info()!=Success) {
//        std::cout << "decomposition failed" << std::endl;
//    }
//    
//    clockInverse.Stop();
//    std::cout << "compute inverse of HessianCw: " << clockInverse.GetTotal() << std::endl;
//    
//    for(unsigned int step = 0; step < noOfSteps; ++step) {
//        std::cout << "step " << step << std::endl;
//        
//        // write current displacement and warped moving images
//        if(verboseResults) {
//            VectorCoefficientImageType::Pointer displacementField = createDisplacementField(weights, controlPointsImage, fixedImage, insideRegion);
//            saveResults(movingImage, displacementField, step, "GN");
//        }
//        
//        // take next step
////        MatrixXf J = MatrixXf::Zero(totalNumberOfPixels,Dimension*totalControlPoints);
////        SparseMatrix<CoeffType> J(totalNumberOfPixels,Dimension*totalControlPoints);
//        MatrixXf gradCw = gradV(alpha, lam, Bmatrix, S, weights, controlPointsImage, fixedImage, movingImage, insideRegion, ssds(step), Jfixed);
////        SparseMatrix<CoeffType> HessianCw = hessV(alpha, lam, J, S);
//        
//        weights = weights - solver.solve(gradCw);
//        if(solver.info()!=Success) {
//            std::cout <<  "solving failed" << std::endl;
//        }
//        
////        MatrixXf HessianCwInv = HessianCw.inverse();
////        MatrixXf HessianCwInv = pinv(HessianCw);
////        weights = weights - HessianCwInv*gradCw;
//        
//        weightsAll.col(step) = weights;
//    }
//
//    std::ostringstream ssdsFilename;
//    ssdsFilename << outputName << "_ssds.txt";
//    std::ofstream ssdsFile;
//    ssdsFile.open (ssdsFilename.str().c_str());
//    ssdsFile << ssds;
//    ssdsFile.close();
//
//    std::ostringstream weightsAllFilename;
//    weightsAllFilename << outputName << "_weightsAll.txt";
//    saveMatrixToFile(weightsAll, weightsAllFilename.str().c_str());
//    
//    clock.Stop();
//    std::cout << "Gauss-Newton (with preperation): " << clock.GetTotal() << std::endl;
    
    // Hamiltonian Monte Carlo
    
    itk::TimeProbe clockHMC;
    clockHMC.Start();
    
    weights = MatrixXf::Zero(Dimension*totalControlPoints, 1);
    
//    std::random_device rd;
//    std::mt19937 gen(rd());
    boost::mt19937 gen(std::time(0));
//    std::uniform_real_distribution< float > runif(0.0,1.0);
    boost::uniform_real<> runif(0.0,1.0);
//    std::normal_distribution<> rnorm(0.0,Kstd);
    boost::normal_distribution<> rnorm(0.0,Kstd);
    
    weightsAll = MatrixXf::Zero(Dimension*totalControlPoints, T);
    ssds = VectorXf::Zero(T);
    
    itk::TimeProbe clockSVD;
    clockSVD.Start();

//    MatrixXf HessianCwInv = solver.solve(MatrixXf::Identity(HessianCw.rows(),HessianCw.cols()));
////    JacobiSVD<MatrixXf> svd(HessianCwInv, ComputeFullU);
////    VectorXf diagonal = svd.singularValues();
////    std::cout << "smallest eigenvalue = " << diagonal.minCoeff() << " larges eigenvalue = " << diagonal.maxCoeff() << std::endl;
////    for(unsigned int i = 0; i < diagonal.size(); ++i)
////        diagonal(i) = std::sqrt(diagonal(i));
////    MatrixXf A = svd.matrixU()*diagonal;
//    Eigen::LLT<MatrixXf> cholesky(HessianCwInv);
//    MatrixXf A = cholesky.matrixL();
//    saveMatrixToFile(HessianCw,"HessianCw.txt");
    Eigen::SimplicialLLT< SparseMatrix<CoeffType> > cholesky;
    Eigen::SparseLU< SparseMatrix<CoeffType> > solverLt;
//    if(hessianProposal) {
//        cholesky.compute(HessianCw);
//        solverLt.compute(cholesky.matrixU());
//    }
//    MatrixXf invMatrixU = solverLt.solve(MatrixXf::Identity(HessianCw.rows(),HessianCw.cols()));
//    saveMatrixToFile(invMatrixU,"invMatrixU.txt");
    
    clockSVD.Stop();
    std::cout << "Compute matrix decomposition to sample form multivariate Gaussian: " << clockSVD.GetTotal() << std::endl;
    
    for(unsigned int step = 0; step < T; ++step) {
        
        std::cout << "step: " << step << std::endl;
        
        // write current displacement and warped moving images
        if(verboseResults) {
            VectorCoefficientImageType::Pointer displacementField = createDisplacementField(weights, controlPointsImage, fixedImage, insideRegion);
            saveResults(movingImage, displacementField, step, "HMC");
        }
    
        // Initialize position and momentum vector
        MatrixXf cur_q = weights;
        MatrixXf cur_p(Dimension*totalControlPoints, 1);
        for(unsigned int i = 0; i < cur_p.size(); ++i)
            cur_p(i,0) = rnorm(gen);
        // transform it so that we sample form a multivariate normal with zero mean and HessianCWInv covariance
//        saveMatrixToFile(cur_p, "cur_p_before.txt");
//        saveMatrixToFile(invMatrixU*cur_p, "invMatrixU*cur_p.txt");
        if(hessianProposal) {
            cur_p = solverLt.solve(cur_p);
            if(solverLt.info()!=Success)
                std::cout <<  "solving failed" << std::endl;
        }
//        saveMatrixToFile(cur_p, "cur_p_after.txt");
        
        // Make a half step for momentum at the beginning
        float cur_ssd = 0;
        //        MatrixXf cur_J = MatrixXf::Zero(totalNumberOfPixels,Dimension*totalControlPoints);
//        SparseMatrix<CoeffType> cur_J(totalNumberOfPixels,Dimension*totalControlPoints);
        MatrixXf cur_gradCw = gradV(alpha, lam, Bmatrix, S, cur_q, controlPointsImage, fixedImage, movingImage, insideRegion, cur_ssd, Jfixed);
        MatrixXf p = cur_p - epsilon * cur_gradCw / 2.0;
        
        // Alternate full steps for position and momentum
        MatrixXf q = cur_q;
        for(unsigned int i = 0; i < L; ++i) {
            // Make a full step for the position
            q = q + epsilon * p;
            
            // Make a full step for the momentum, except at end of trajectory
            if (i!=(L-1)) {
                float ssd = 0;
                //        MatrixXf J = MatrixXf::Zero(totalNumberOfPixels,Dimension*totalControlPoints);
//                SparseMatrix<CoeffType> J(totalNumberOfPixels,Dimension*totalControlPoints);
                MatrixXf gradCw = gradV(alpha, lam, Bmatrix, S, q, controlPointsImage, fixedImage, movingImage, insideRegion, ssd, Jfixed);
                p = p - epsilon * gradCw;
            }
        }
        // New proposed position
        MatrixXf pro_q = q;
        
        // Make a half step for momentum at the end
        float pro_ssd = 0;
        //        MatrixXf pro_J = MatrixXf::Zero(totalNumberOfPixels,Dimension*totalControlPoints);
//        SparseMatrix<CoeffType> pro_J(totalNumberOfPixels,Dimension*totalControlPoints);
        MatrixXf pro_gradCw = gradV(alpha, lam, Bmatrix, S, pro_q, controlPointsImage, fixedImage, movingImage, insideRegion, pro_ssd, Jfixed);
        p = p - epsilon * pro_gradCw / 2.0;
        
        // Negate momentum at end of trajectory to make the proposal symmetric
        p = -p;
        
        // New proposed momentum
        MatrixXf pro_p = p;
        
        // Evaluate potential and kinetic energies at start and end of trajectory
        float cur_V = potentialV(cur_q, alpha, lam, S, cur_ssd);
        float cur_K = (cur_p.transpose()*cur_p)(0,0) / 2.0;
        float pro_V = potentialV(pro_q, alpha, lam, S, pro_ssd);
        float pro_K = (pro_p.transpose()*pro_p)(0,0) / 2.0;
        std::cout << "cur_V " << cur_V << " - pro_V " << pro_V << " cur_K " << cur_K << " pro_K " << pro_K << " = " << cur_V-pro_V+cur_K-pro_K << std::endl;
        
        // compare current with proposed values
        double acceptanceProb = std::exp(cur_V-pro_V+cur_K-pro_K);
        std::cout << "acceptanceProb = " << acceptanceProb << std::endl;
        double uniformNumber = runif(gen);
        std::cout << "uniformNumber = " << uniformNumber << std::endl;
        bool accept = uniformNumber < acceptanceProb;
        if(accept) {
            std::cout << "accept" << std::endl;
            ssds(step) = pro_ssd;
            weights = pro_q;
            weightsAll.col(step) = pro_q;
        }
        else {
            std::cout << "reject" << std::endl;
            ssds(step) = cur_ssd;
            weights = cur_q;
            weightsAll.col(step) = cur_q;
        }
        
    }
    
    std::ostringstream ssdsHMCFilename;
    ssdsHMCFilename << outputName << "_ssds_HMC.txt";
    std::ofstream ssdsHMCFile;
    ssdsHMCFile.open (ssdsHMCFilename.str().c_str());
    ssdsHMCFile << ssds;
    ssdsHMCFile.close();
    
    std::ostringstream weightsAllHMCFilename;
    weightsAllHMCFilename << outputName << "_weightsAll_HMC.txt";
    saveMatrixToFile(weightsAll, weightsAllHMCFilename.str().c_str());
    
    clockHMC.Stop();
    std::cout << "HMC: " << clockHMC.GetTotal() << std::endl;
    
    return EXIT_SUCCESS;
}
