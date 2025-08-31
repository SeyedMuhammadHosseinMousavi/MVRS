using System;
using System.Globalization;
using System.IO;
using System.Text;
using Microsoft.Kinect;
using ClosedXML.Excel;

namespace KinectDataProcessing
{
    class Program
    {
        static KinectSensor sensor;
        static StreamWriter textWriter;
        static StreamWriter csvWriter;
        static XLWorkbook workbook;
        static IXLWorksheet worksheet;
        static int excelRow = 2; // Start writing data from the second row (first row for headers)

        // --- FIX: track absolute start time based on the first frame's RelativeTime ---
        static bool haveStartTime = false;
        static DateTime sensorStartUtc; // absolute UTC time for RelativeTime = 0

        static void Main(string[] args)
        {
            string directoryPath = @"C:\KinectData";
            Directory.CreateDirectory(directoryPath); // Ensure the directory exists

            string textFilePath = Path.Combine(directoryPath, "JointData.txt");
            string excelFilePath = Path.Combine(directoryPath, "JointData.xlsx");
            string csvFilePath = Path.Combine(directoryPath, "JointData.csv");

            workbook = new XLWorkbook();
            worksheet = workbook.AddWorksheet("Joint Data");
            SetupExcelHeaders();

            using (textWriter = new StreamWriter(textFilePath))
            using (csvWriter = new StreamWriter(csvFilePath))
            {
                // Write headers for the text and CSV file
                WriteHeaders(textWriter);
                WriteHeaders(csvWriter);

                sensor = KinectSensor.GetDefault();
                if (sensor != null)
                {
                    sensor.Open();
                    BodyFrameReader reader = sensor.BodyFrameSource.OpenReader();
                    reader.FrameArrived += Reader_FrameArrived;

                    Console.WriteLine("Collecting data. Press any key to exit...");
                    Console.WriteLine($"Data is being saved to {textFilePath}, {excelFilePath}, and {csvFilePath}");

                    Console.ReadKey();

                    reader.Dispose();
                    workbook.SaveAs(excelFilePath);
                    sensor.Close();
                }
                else
                {
                    Console.WriteLine("No Kinect sensor found.");
                }
            }
        }

        static void SetupExcelHeaders()
        {
            worksheet.Cell("A1").Value = "Timestamp";
            int col = 2;
            foreach (JointType joint in Enum.GetValues(typeof(JointType)))
            {
                worksheet.Cell(1, col++).Value = $"{joint}_PosX";
                worksheet.Cell(1, col++).Value = $"{joint}_PosY";
                worksheet.Cell(1, col++).Value = $"{joint}_PosZ";
                worksheet.Cell(1, col++).Value = $"{joint}_RotX";
                worksheet.Cell(1, col++).Value = $"{joint}_RotY";
                worksheet.Cell(1, col++).Value = $"{joint}_RotZ";
                worksheet.Cell(1, col++).Value = $"{joint}_RotW";
            }
        }

        static void WriteHeaders(StreamWriter writer)
        {
            StringBuilder headers = new StringBuilder("Timestamp, ");
            foreach (JointType joint in Enum.GetValues(typeof(JointType)))
            {
                headers.AppendFormat("{0}_PosX, {0}_PosY, {0}_PosZ, {0}_RotX, {0}_RotY, {0}_RotZ, {0}_RotW, ", joint);
            }
            writer.WriteLine(headers.ToString().TrimEnd(',', ' '));
        }

        static void Reader_FrameArrived(object sender, BodyFrameArrivedEventArgs e)
        {
            using (BodyFrame frame = e.FrameReference.AcquireFrame())
            {
                if (frame != null)
                {
                    // --- FIX: establish absolute start time using the first frame's RelativeTime ---
                    if (!haveStartTime)
                    {
                        sensorStartUtc = DateTime.UtcNow - frame.RelativeTime;
                        haveStartTime = true;
                    }

                    // frame.RelativeTime is a TimeSpan since sensor started streaming
                    DateTime absoluteUtc = sensorStartUtc + frame.RelativeTime;
                    // ISO 8601 (UTC) for files; also make a local display string if you prefer
                    string timestampIsoUtc = absoluteUtc.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);

                    Body[] bodies = new Body[sensor.BodyFrameSource.BodyCount];
                    frame.GetAndRefreshBodyData(bodies);

                    foreach (Body body in bodies)
                    {
                        if (body.IsTracked)
                        {
                            // --- FIX: use Kinect-derived timestamp, not DateTime.Now ---
                            StringBuilder line = new StringBuilder(timestampIsoUtc + ", ");
                            int col = 2;

                            // also write timestamp into Excel column A
                            worksheet.Cell(excelRow, 1).Value = absoluteUtc;         // store as DateTime
                            worksheet.Cell(excelRow, 1).Style.DateFormat.Format = "yyyy-mm-dd hh:mm:ss.000";

                            foreach (JointType jointType in Enum.GetValues(typeof(JointType)))
                            {
                                Joint joint = body.Joints[jointType];
                                JointOrientation orientation = body.JointOrientations[jointType];

                                string jointData = string.Format(CultureInfo.InvariantCulture,
                                    "{0:F6}, {1:F6}, {2:F6}, {3:F6}, {4:F6}, {5:F6}, {6:F6}, ",
                                    joint.Position.X, joint.Position.Y, joint.Position.Z,
                                    orientation.Orientation.X, orientation.Orientation.Y,
                                    orientation.Orientation.Z, orientation.Orientation.W);

                                line.Append(jointData);

                                worksheet.Cell(excelRow, col++).Value = joint.Position.X;
                                worksheet.Cell(excelRow, col++).Value = joint.Position.Y;
                                worksheet.Cell(excelRow, col++).Value = joint.Position.Z;
                                worksheet.Cell(excelRow, col++).Value = orientation.Orientation.X;
                                worksheet.Cell(excelRow, col++).Value = orientation.Orientation.Y;
                                worksheet.Cell(excelRow, col++).Value = orientation.Orientation.Z;
                                worksheet.Cell(excelRow, col++).Value = orientation.Orientation.W;
                            }

                            textWriter.WriteLine(line.ToString());
                            csvWriter.WriteLine(line.ToString());
                            Console.WriteLine(line.ToString()); // Real-time feedback
                            excelRow++;
                        }
                    }
                }
            }
        }
    }
}
