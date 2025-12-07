class ActuatorSystem:
    def ejecutar(self, acciones):
        if not acciones:
            print("   ✅ Sistema en equilibrio. Standby.")
            return

        print("   ⚠️  ACCIONES CORRECTIVAS INICIADAS:")
        for accion in acciones:
            # Aquí iría el código real: GPIO.output(PIN_BOMBA, HIGH)
            print(f"      >> EJECUTANDO: {accion}")