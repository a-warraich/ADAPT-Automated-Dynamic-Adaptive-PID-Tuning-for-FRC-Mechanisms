package frc.robot.subsystems;

import edu.wpi.first.wpilibj2.command.SubsystemBase;

import com.revrobotics.spark.SparkMax;
import com.revrobotics.spark.SparkLowLevel.MotorType;
import com.revrobotics.spark.SparkBase;
import com.revrobotics.spark.SparkRelativeEncoder;
import com.revrobotics.spark.config.SparkMaxConfig;
import com.revrobotics.RelativeEncoder;
import com.revrobotics.sim.SparkMaxSim;

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.controller.PIDController;
import edu.wpi.first.math.system.LinearSystem;
import edu.wpi.first.math.system.plant.DCMotor;
import edu.wpi.first.math.system.plant.LinearSystemId;
import edu.wpi.first.math.numbers.N1;

import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj.simulation.FlywheelSim;
import edu.wpi.first.wpilibj.simulation.RoboRioSim;
import edu.wpi.first.wpilibj.simulation.BatterySim;

import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.DoubleSubscriber;
import edu.wpi.first.networktables.BooleanSubscriber;
import edu.wpi.first.networktables.DoubleArrayPublisher;
import edu.wpi.first.networktables.BooleanPublisher;

public class GAPIDSparkMaxSubsystem extends SubsystemBase {

    private static final int kMotorId = 1;

    private final SparkMax m_motor;
    private final RelativeEncoder m_encoder;
    private final SparkMaxConfig m_config;

    private final PIDController m_pid;
    private double m_lastPidVolts = 0.0;

    private final SparkMaxSim m_motorSim;
    private final FlywheelSim m_flywheelSim;
    private final NetworkTable table;

    private final DoubleSubscriber kpSub;
    private final DoubleSubscriber kiSub;
    private final DoubleSubscriber kdSub;
    private final DoubleSubscriber setpointSub;
    private final BooleanSubscriber startSub;

    private final BooleanPublisher donePub;
    private final DoubleArrayPublisher errorPub;
    private final DoubleArrayPublisher outputPub;

    private static final int SAMPLES = 150;
    private final double[] errorBuf = new double[SAMPLES];
    private final double[] outputBuf = new double[SAMPLES];
    private int index = 0;
    private boolean running = false;
    private double desiredRPM = 0.0;
    private double startTime = 0.0;

    public GAPIDSparkMaxSubsystem() {

        m_motor = new SparkMax(kMotorId, MotorType.kBrushless);
        m_encoder = m_motor.getEncoder();
        m_config = new SparkMaxConfig();

        m_config.inverted(false);
        m_config.idleMode(SparkMaxConfig.IdleMode.kBrake);
        m_config.encoder.velocityConversionFactor(1.0);

        m_motor.configure(
                m_config,
                SparkBase.ResetMode.kResetSafeParameters,
                SparkBase.PersistMode.kNoPersistParameters
        );

        m_pid = new PIDController(0.0, 0.0, 0.0);
        m_pid.setTolerance(50.0);

        DCMotor neo = DCMotor.getNEO(1);
        LinearSystem<N1, N1, N1> plant =
                LinearSystemId.createFlywheelSystem(neo, 1.0, 0.002);

        m_flywheelSim = new FlywheelSim(plant, neo);
        m_motorSim = new SparkMaxSim(m_motor, neo);

        table = NetworkTableInstance.getDefault().getTable("GA_PID");

        kpSub = table.getDoubleTopic("kp").subscribe(0.0);
        kiSub = table.getDoubleTopic("ki").subscribe(0.0);
        kdSub = table.getDoubleTopic("kd").subscribe(0.0);
        setpointSub = table.getDoubleTopic("setpoint").subscribe(0.0);
        startSub = table.getBooleanTopic("startTest").subscribe(false);

        donePub = table.getBooleanTopic("testDone").publish();
        errorPub = table.getDoubleArrayTopic("errors").publish();
        outputPub = table.getDoubleArrayTopic("outputs").publish();

        donePub.set(false);
    }

    @Override
    public void periodic() {
        if (startSub.get() && !running) {
            beginTest();
        }

        if (running) {
            runTest();
        }
    }

    private void beginTest() {
        double kp = kpSub.get();
        double ki = kiSub.get();
        double kd = kdSub.get();
        desiredRPM = setpointSub.get();

        m_pid.setP(kp);
        m_pid.setI(ki);
        m_pid.setD(kd);
        m_pid.reset();

        index = 0;
        for (int i = 0; i < SAMPLES; i++) {
            errorBuf[i] = 0.0;
            outputBuf[i] = 0.0;
        }

        running = true;
        startTime = Timer.getFPGATimestamp();
        donePub.set(false);

        System.out.println("[GA PID] START kp=" + kp + " ki=" + ki + " kd=" + kd
                + "  SP=" + desiredRPM);
    }

    private void runTest() {
        double currentRPM = m_encoder.getVelocity();
        double pidVolts = m_pid.calculate(currentRPM, desiredRPM);
        pidVolts = MathUtil.clamp(pidVolts, -12.0, 12.0);

        m_lastPidVolts = pidVolts;
        m_motor.setVoltage(pidVolts);

        double err = desiredRPM - currentRPM;

        if (index < SAMPLES) {
            errorBuf[index] = err;
            outputBuf[index] = currentRPM;
            index++;
        }

        if (index >= SAMPLES || (Timer.getFPGATimestamp() - startTime) > 5.0) {
            finishTest();
        }
    }

    private void finishTest() {
        running = false;

        double[] e = new double[index];
        double[] o = new double[index];
        System.arraycopy(errorBuf, 0, e, 0, index);
        System.arraycopy(outputBuf, 0, o, 0, index);

        errorPub.set(e);
        outputPub.set(o);
        donePub.set(true);

        System.out.println("[GA PID] DONE (" + index + " samples)");
    }

    @Override
    public void simulationPeriodic() {
        double battery = RoboRioSim.getVInVoltage();

        m_flywheelSim.setInputVoltage(m_lastPidVolts);
        m_flywheelSim.update(0.02);

        double rpm = m_flywheelSim.getAngularVelocityRPM();
        m_motorSim.iterate(rpm, battery, 0.02);

        double current = m_flywheelSim.getCurrentDrawAmps();
        RoboRioSim.setVInVoltage(
                BatterySim.calculateDefaultBatteryLoadedVoltage(current)
        );
    }

    public double getVelocityRPM() {
        return m_encoder.getVelocity();
    }
}
